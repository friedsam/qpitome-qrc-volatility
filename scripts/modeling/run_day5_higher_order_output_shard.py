#!/usr/bin/env python3
"""Run the final standardized higher-order output assay for the frozen day-5 QRC."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.qrc.local_detuning_reservoir import evolve_local_detuning_states
from qpitome_qrc.qrc.rydberg_reservoir import precompute

REPO = Path(__file__).resolve().parents[2]
BASE_PATH = REPO / "scripts" / "modeling" / "run_day5_residualized_rydberg_crossfit_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_crossfit_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load helpers from {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

RIDGE_ALPHA = 10.0
OFFSET_L2 = 100.0
RNG_SEED = 20260711


def exact_output_blocks(states: np.ndarray, occ_bits: np.ndarray) -> dict[str, np.ndarray]:
    """Extract predeclared full-counting and pooled third-order observables."""
    probs = np.abs(states) ** 2
    n_occ = occ_bits.sum(axis=1).astype(int)
    n_atoms = occ_bits.shape[1]

    histogram = np.column_stack([
        probs[:, n_occ == k].sum(axis=1) for k in range(n_atoms)
    ])

    adjacent_count = np.sum(occ_bits[:, :-1] * occ_bits[:, 1:], axis=1)
    complementary_pairs = [(i, i + 1) for i in range(0, n_atoms, 2)]
    complementary_count = sum(occ_bits[:, i] * occ_bits[:, j] for i, j in complementary_pairs)
    domain_walls = np.sum(np.abs(np.diff(occ_bits, axis=1)), axis=1)
    blockade = np.column_stack([
        probs[:, n_occ <= 1].sum(axis=1),
        probs @ adjacent_count,
        probs[:, adjacent_count > 0].sum(axis=1),
        probs @ complementary_count,
        probs[:, complementary_count > 0].sum(axis=1),
        probs @ domain_walls,
    ])

    means = probs @ occ_bits
    pair = np.empty((len(states), n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
        for j in range(n_atoms):
            pair[:, i, j] = probs @ (occ_bits[:, i] * occ_bits[:, j])

    pooled = np.zeros((len(states), 3), dtype=float)
    counts = np.zeros(3, dtype=int)
    for i, j, k in combinations(range(n_atoms), 3):
        triple = probs @ (occ_bits[:, i] * occ_bits[:, j] * occ_bits[:, k])
        cumulant = (
            triple
            - pair[:, i, j] * means[:, k]
            - pair[:, i, k] * means[:, j]
            - pair[:, j, k] * means[:, i]
            + 2.0 * means[:, i] * means[:, j] * means[:, k]
        )
        span = k - i
        bucket = 0 if span == 2 else (1 if span <= 5 else 2)
        pooled[:, bucket] += cumulant
        counts[bucket] += 1
    pooled /= counts[None, :]

    return {
        "count_histogram": histogram,
        "blockade_patterns": blockade,
        "pooled_triples": pooled,
        "higher_order_all": np.column_stack([histogram, blockade, pooled]),
    }


def run_shard(frame: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    config = base.base.assay.rydberg_config()
    pre = precompute(config.reservoir)
    eligible = base.base.assay.eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))
    selected = [group for pos, group in enumerate(fold_groups) if pos % num_shards == shard_index]
    records: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(selected, start=1):
        train = frame[frame["landmark_date"] < cluster_start].copy()
        test_indices = test_group.index.to_numpy(int)
        combined = pd.concat([train, frame.loc[test_indices]], axis=0)

        patterns = base.base.assay.differential_patterns(
            train[base.base.assay.STATIC].to_numpy(float),
            combined[base.base.assay.STATIC].to_numpy(float),
        )
        states = evolve_local_detuning_states(patterns, config)
        blocks = exact_output_blocks(states, pre.occ_bits)

        d1_train = train[base.base.assay.D1].to_numpy(float)
        d1_test = frame.loc[test_indices, base.base.assay.D1].to_numpy(float)
        y_train = train["y_recovery"].to_numpy(int)
        X_train = base.base.d1_basis(d1_train, "quadratic")
        X_test = base.base.d1_basis(d1_test, "quadratic")

        cf_positions, cf_logits = base.historical_crossfit_d1_logits(train)
        if len(cf_positions) < base.MIN_CORRECTION_TRAIN:
            raise RuntimeError(f"Insufficient correction rows for {cluster_start}")

        d1_model = base.base.assay.logistic_pipeline(1.0)
        d1_model.fit(d1_train, y_train)
        p_test_d1 = d1_model.predict_proba(d1_test)[:, 1]
        offset_test = base.base.assay.logit(p_test_d1)

        predictions: dict[str, np.ndarray] = {"D1": p_test_d1}
        rng = np.random.default_rng(RNG_SEED + int(pd.Timestamp(cluster_start).value % 2**31))

        for block_name, block in blocks.items():
            H_train = block[: len(train)]
            H_test = block[len(train) :]
            R_train, R_test = base.base.residualize_train_test(
                X_train, H_train, X_test, H_test, RIDGE_ALPHA
            )
            values = []
            for row_number in range(len(test_indices)):
                values.append(base.base.assay.fit_offset_predict(
                    R_train[cf_positions], y_train[cf_positions], R_test[[row_number]],
                    cf_logits, float(offset_test[row_number]), OFFSET_L2,
                ))
            predictions[f"higher_{block_name}"] = np.asarray(values)

            gaussian_train = rng.normal(size=H_train.shape)
            gaussian_test = rng.normal(size=H_test.shape)
            null_values = []
            for row_number in range(len(test_indices)):
                null_values.append(base.base.assay.fit_offset_predict(
                    gaussian_train[cf_positions], y_train[cf_positions], gaussian_test[[row_number]],
                    cf_logits, float(offset_test[row_number]), OFFSET_L2,
                ))
            predictions[f"gaussian_{block_name}"] = np.asarray(null_values)

        for local_row, row_index in enumerate(test_indices):
            row = frame.loc[row_index]
            record = {
                "row_id": int(row_index),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "cluster_start": cluster_start,
                "y": int(row["y_recovery"]),
                "shard_index": shard_index,
            }
            for name, values in predictions.items():
                record[name] = float(values[local_row])
            records.append(record)

        print(
            f"shard {shard_index}/{num_shards}: fold {fold_number}/{len(selected)} "
            f"cluster_start={pd.Timestamp(cluster_start).date()} train={len(train)} test={len(test_indices)}",
            flush=True,
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    frame = base.base.assay.load_frame(args.path_panel, args.clusters)
    predictions = run_shard(frame, args.shard_index, args.num_shards)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / f"predictions_shard_{args.shard_index}.csv", index=False)
    (args.outdir / f"manifest_shard_{args.shard_index}.json").write_text(json.dumps({
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "ridge_alpha": RIDGE_ALPHA,
        "offset_l2": OFFSET_L2,
        "blocks": ["count_histogram", "blockade_patterns", "pooled_triples", "higher_order_all"],
    }, indent=2))


if __name__ == "__main__":
    main()
