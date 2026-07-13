#!/usr/bin/env python3
"""Test whether Rydberg outputs contain signal beyond D1.

For each historical outer fold, this script:
1. generates the frozen differential-pair spatial Rydberg outputs;
2. residualizes each output block against D1 using cross-fitted Ridge models;
3. freezes the D1 logit;
4. fits only a strongly regularized logistic correction from residual outputs.

Work is sharded by unique cluster start, so rows sharing one training cutoff reuse the
same Rydberg simulation and residualizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.logistic_offset import logit
from qpitome_qrc.day5.features import split_blocks
from qpitome_qrc.day5.protocol import (
    D1,
    STATIC,
    differential_patterns,
    eligible_rows,
    load_frame,
    rydberg_config,
)
from qpitome_qrc.evaluation.binary import fit_offset_predict, logistic_pipeline
from qpitome_qrc.evaluation.residualization import (
    DEFAULT_N_SPLITS,
    d1_basis,
    fit_residualizer,
    residual_diagnostics,
    residualize_train_test,
)
from qpitome_qrc.qrc.local_detuning_reservoir import build_local_detuning_feature_matrix

BLOCKS = ("occupations", "all_raw", "occ_plus_connected")
RESIDUALIZERS = ("linear", "quadratic")
RIDGE_ALPHAS = (1.0, 10.0)
OFFSET_L2 = (100.0, 1000.0)
N_SPLITS = DEFAULT_N_SPLITS


def run_shard(
    frame: pd.DataFrame,
    shard_index: int,
    num_shards: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = rydberg_config()
    eligible = eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))
    selected = [group for position, group in enumerate(fold_groups) if position % num_shards == shard_index]

    predictions: list[dict] = []
    diagnostics: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(selected, start=1):
        train = frame[frame["landmark_date"] < cluster_start]
        test_indices = test_group.index.to_numpy(int)
        combined = pd.concat([train, frame.loc[test_indices]], axis=0)

        patterns = differential_patterns(
            train[STATIC].to_numpy(float),
            combined[STATIC].to_numpy(float),
        )
        features = build_local_detuning_feature_matrix(patterns, config)
        blocks = split_blocks(features)

        d1_train = train[D1].to_numpy(float)
        d1_test = frame.loc[test_indices, D1].to_numpy(float)
        y_train = train["y_recovery"].to_numpy(int)
        d1_model = logistic_pipeline(1.0)
        d1_model.fit(d1_train, y_train)
        p_train = d1_model.predict_proba(d1_train)[:, 1]
        p_test = d1_model.predict_proba(d1_test)[:, 1]
        offset_train = logit(p_train)
        offset_test = logit(p_test)

        fold_predictions: dict[str, np.ndarray] = {"D1": p_test}

        for block_name in BLOCKS:
            block = blocks[block_name]
            H_train = block[: len(train)]
            H_test = block[len(train) :]
            for residualizer in RESIDUALIZERS:
                X_train = d1_basis(d1_train, residualizer)
                X_test = d1_basis(d1_test, residualizer)
                for ridge_alpha in RIDGE_ALPHAS:
                    R_train, R_test = residualize_train_test(
                        X_train, H_train, X_test, H_test, ridge_alpha
                    )
                    diag = residual_diagnostics(H_train, R_train)
                    diagnostics.append({
                        "cluster_start": cluster_start,
                        "block": block_name,
                        "residualizer": residualizer,
                        "ridge_alpha": ridge_alpha,
                        "n_train": int(len(train)),
                        "n_test": int(len(test_indices)),
                        **diag,
                    })
                    for l2 in OFFSET_L2:
                        name = f"offset_resid_{block_name}_{residualizer}_ridge{ridge_alpha:g}_l2{int(l2)}"
                        values = []
                        for row_number in range(len(test_indices)):
                            values.append(
                                fit_offset_predict(
                                    R_train,
                                    y_train,
                                    R_test[[row_number]],
                                    offset_train,
                                    float(offset_test[row_number]),
                                    l2,
                                )
                            )
                        fold_predictions[name] = np.asarray(values, dtype=float)

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
            for name, values in fold_predictions.items():
                record[name] = float(values[local_row])
            predictions.append(record)

        print(
            f"shard {shard_index}/{num_shards}: fold {fold_number}/{len(selected)} "
            f"cluster_start={pd.Timestamp(cluster_start).date()} train={len(train)} test={len(test_indices)}",
            flush=True,
        )

    return pd.DataFrame(predictions), pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")

    frame = load_frame(args.path_panel, args.clusters)
    predictions, diagnostics = run_shard(frame, args.shard_index, args.num_shards)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / f"predictions_shard_{args.shard_index}.csv", index=False)
    diagnostics.to_csv(args.outdir / f"residual_diagnostics_shard_{args.shard_index}.csv", index=False)
    manifest = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "blocks": BLOCKS,
        "residualizers": RESIDUALIZERS,
        "ridge_alphas": RIDGE_ALPHAS,
        "offset_l2": OFFSET_L2,
        "n_predictions": int(len(predictions)),
    }
    (args.outdir / f"manifest_shard_{args.shard_index}.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved shard {args.shard_index} to {args.outdir}")


if __name__ == "__main__":
    main()
