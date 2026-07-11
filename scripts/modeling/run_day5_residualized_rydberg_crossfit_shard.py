#!/usr/bin/env python3
"""Frozen residualized-Rydberg candidate with historical cross-fitted D1 offsets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE_PATH = REPO / "scripts" / "modeling" / "run_day5_residualized_rydberg_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_residualized_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load helpers from {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

BLOCK = "occupations"
RESIDUALIZER = "quadratic"
RIDGE_ALPHA = 10.0
OFFSET_L2 = 100.0


def historical_crossfit_d1_logits(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return indices and strictly historical out-of-fold D1 logits for train rows."""
    logits = np.full(len(train), np.nan, dtype=float)
    for cluster_start, group in train.groupby("cluster_start", sort=True):
        valid_positions = train.index.get_indexer(group.index)
        prior = train[train["landmark_date"] < cluster_start]
        if len(prior) < base.assay.MIN_TRAIN or prior["y_recovery"].nunique() < 2:
            continue
        model = base.assay.logistic_pipeline(1.0)
        model.fit(prior[base.assay.D1].to_numpy(float), prior["y_recovery"].to_numpy(int))
        p = model.predict_proba(group[base.assay.D1].to_numpy(float))[:, 1]
        logits[valid_positions] = base.assay.logit(p)
    valid = np.isfinite(logits)
    return np.flatnonzero(valid), logits[valid]


def run_shard(frame: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    config = base.assay.rydberg_config()
    eligible = base.assay.eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))
    selected = [group for pos, group in enumerate(fold_groups) if pos % num_shards == shard_index]
    records: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(selected, start=1):
        train = frame[frame["landmark_date"] < cluster_start].copy()
        test_indices = test_group.index.to_numpy(int)
        combined = pd.concat([train, frame.loc[test_indices]], axis=0)

        patterns = base.assay.differential_patterns(
            train[base.assay.STATIC].to_numpy(float),
            combined[base.assay.STATIC].to_numpy(float),
        )
        features = base.assay.build_local_detuning_feature_matrix(patterns, config)
        blocks = base.assay.split_blocks(features)
        H_train = blocks[BLOCK][: len(train)]
        H_test = blocks[BLOCK][len(train) :]

        d1_train = train[base.assay.D1].to_numpy(float)
        d1_test = frame.loc[test_indices, base.assay.D1].to_numpy(float)
        y_train = train["y_recovery"].to_numpy(int)

        X_train = base.d1_basis(d1_train, RESIDUALIZER)
        X_test = base.d1_basis(d1_test, RESIDUALIZER)
        R_train, R_test = base.residualize_train_test(
            X_train, H_train, X_test, H_test, RIDGE_ALPHA
        )

        cf_positions, cf_logits = historical_crossfit_d1_logits(train)
        if len(cf_positions) < base.assay.MIN_TRAIN or np.unique(y_train[cf_positions]).size < 2:
            raise RuntimeError(f"Insufficient cross-fitted rows for cluster {cluster_start}")

        full_d1 = base.assay.logistic_pipeline(1.0)
        full_d1.fit(d1_train, y_train)
        p_test_d1 = full_d1.predict_proba(d1_test)[:, 1]
        offset_test = base.assay.logit(p_test_d1)

        p_test_corrected = []
        for row_number in range(len(test_indices)):
            p_test_corrected.append(
                base.assay.fit_offset_predict(
                    R_train[cf_positions],
                    y_train[cf_positions],
                    R_test[[row_number]],
                    cf_logits,
                    float(offset_test[row_number]),
                    OFFSET_L2,
                )
            )

        for local_row, row_index in enumerate(test_indices):
            row = frame.loc[row_index]
            records.append({
                "row_id": int(row_index),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "cluster_start": cluster_start,
                "y": int(row["y_recovery"]),
                "D1": float(p_test_d1[local_row]),
                "crossfit_resid_occupations": float(p_test_corrected[local_row]),
                "n_correction_train": int(len(cf_positions)),
                "shard_index": shard_index,
            })

        print(
            f"shard {shard_index}/{num_shards}: fold {fold_number}/{len(selected)} "
            f"cluster_start={pd.Timestamp(cluster_start).date()} train={len(train)} "
            f"correction_train={len(cf_positions)} test={len(test_indices)}",
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

    frame = base.assay.load_frame(args.path_panel, args.clusters)
    predictions = run_shard(frame, args.shard_index, args.num_shards)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / f"predictions_shard_{args.shard_index}.csv", index=False)
    manifest = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "block": BLOCK,
        "residualizer": RESIDUALIZER,
        "ridge_alpha": RIDGE_ALPHA,
        "offset_l2": OFFSET_L2,
        "n_predictions": int(len(predictions)),
    }
    (args.outdir / f"manifest_shard_{args.shard_index}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
