#!/usr/bin/env python3
"""Frozen residualized-Rydberg candidate with historical cross-fitted D1 offsets."""

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
from qpitome_qrc.evaluation.historical_crossfit import historical_crossfit_d1_logits
from qpitome_qrc.evaluation.residualization import d1_basis, residualize_train_test
from qpitome_qrc.qrc.local_detuning_reservoir import build_local_detuning_feature_matrix

BLOCK = "occupations"
RESIDUALIZER = "quadratic"
RIDGE_ALPHA = 10.0
OFFSET_L2 = 100.0
INNER_CROSSFIT_MIN_TRAIN = 10
MIN_CORRECTION_TRAIN = 20


def run_shard(frame: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    config = rydberg_config()
    eligible = eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))
    selected = [group for pos, group in enumerate(fold_groups) if pos % num_shards == shard_index]
    records: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(selected, start=1):
        train = frame[frame["landmark_date"] < cluster_start].copy()
        test_indices = test_group.index.to_numpy(int)
        combined = pd.concat([train, frame.loc[test_indices]], axis=0)

        patterns = differential_patterns(
            train[STATIC].to_numpy(float),
            combined[STATIC].to_numpy(float),
        )
        features = build_local_detuning_feature_matrix(patterns, config)
        blocks = split_blocks(features)
        H_train = blocks[BLOCK][: len(train)]
        H_test = blocks[BLOCK][len(train) :]

        d1_train = train[D1].to_numpy(float)
        d1_test = frame.loc[test_indices, D1].to_numpy(float)
        y_train = train["y_recovery"].to_numpy(int)

        X_train = d1_basis(d1_train, RESIDUALIZER)
        X_test = d1_basis(d1_test, RESIDUALIZER)
        R_train, R_test = residualize_train_test(
            X_train, H_train, X_test, H_test, RIDGE_ALPHA
        )

        cf_positions, cf_logits = historical_crossfit_d1_logits(
            train,
            min_train=INNER_CROSSFIT_MIN_TRAIN,
        )
        if len(cf_positions) < MIN_CORRECTION_TRAIN or np.unique(y_train[cf_positions]).size < 2:
            raise RuntimeError(
                f"Insufficient cross-fitted rows for cluster {cluster_start}: "
                f"got {len(cf_positions)}, need {MIN_CORRECTION_TRAIN}"
            )

        full_d1 = logistic_pipeline(1.0)
        full_d1.fit(d1_train, y_train)
        p_test_d1 = full_d1.predict_proba(d1_test)[:, 1]
        offset_test = logit(p_test_d1)

        p_test_corrected = []
        for row_number in range(len(test_indices)):
            p_test_corrected.append(
                fit_offset_predict(
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

    frame = load_frame(args.path_panel, args.clusters)
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
        "inner_crossfit_min_train": INNER_CROSSFIT_MIN_TRAIN,
        "min_correction_train": MIN_CORRECTION_TRAIN,
        "n_predictions": int(len(predictions)),
    }
    (args.outdir / f"manifest_shard_{args.shard_index}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
