#!/usr/bin/env python3
"""Classical D1 audit for predeclared non-D1 path-shape input blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from qpitome_qrc.day5.features import PATH_SHAPE_BLOCKS, add_path_shape_features
from qpitome_qrc.day5.protocol import D1, eligible_rows, load_frame
from qpitome_qrc.evaluation.binary import logistic_pipeline
from qpitome_qrc.evaluation.residualization import ridge_pipeline
from qpitome_qrc.evaluation.scoring import proper_score_deltas

# Backward-compatible historical name used by manifests and downstream checks.
BLOCKS = PATH_SHAPE_BLOCKS


def run_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = add_path_shape_features(frame)
    eligible = eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))

    prediction_rows: list[dict] = []
    reconstruction_rows: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(fold_groups, start=1):
        train = frame[frame["landmark_date"] < cluster_start]
        test = frame.loc[test_group.index]
        y_train = train["y_recovery"].to_numpy(int)
        d1_train = train[D1].to_numpy(float)
        d1_test = test[D1].to_numpy(float)

        d1_model = logistic_pipeline(1.0)
        d1_model.fit(d1_train, y_train)
        predictions: dict[str, np.ndarray] = {
            "D1": d1_model.predict_proba(d1_test)[:, 1]
        }

        for block_name, columns in BLOCKS.items():
            H_train = train[columns].to_numpy(float)
            H_test = test[columns].to_numpy(float)

            augmented = logistic_pipeline(1.0)
            augmented.fit(np.column_stack([d1_train, H_train]), y_train)
            predictions[f"D1_plus_{block_name}"] = augmented.predict_proba(
                np.column_stack([d1_test, H_test])
            )[:, 1]

            recon = ridge_pipeline(10.0)
            recon.fit(d1_train, H_train)
            H_pred = recon.predict(d1_test)
            for local_row, row_index in enumerate(test.index):
                for column_index, feature in enumerate(columns):
                    reconstruction_rows.append({
                        "row_id": int(row_index),
                        "cluster_id": test.loc[row_index, "cluster_id"],
                        "block": block_name,
                        "feature": feature,
                        "actual": float(H_test[local_row, column_index]),
                        "predicted_from_D1": float(H_pred[local_row, column_index]),
                    })

        for local_row, row_index in enumerate(test.index):
            row = test.loc[row_index]
            record = {
                "row_id": int(row_index),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "cluster_start": cluster_start,
                "y": int(row["y_recovery"]),
            }
            for name, values in predictions.items():
                record[name] = float(values[local_row])
            prediction_rows.append(record)

        print(
            f"fold {fold_number}/{len(fold_groups)} cluster_start={pd.Timestamp(cluster_start).date()} "
            f"train={len(train)} test={len(test)}",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows).sort_values("row_id").reset_index(drop=True)
    reconstruction = pd.DataFrame(reconstruction_rows)

    recon_summary_rows = []
    for (block, feature), group in reconstruction.groupby(["block", "feature"], sort=True):
        actual = group["actual"].to_numpy(float)
        predicted = group["predicted_from_D1"].to_numpy(float)
        recon_summary_rows.append({
            "block": block,
            "feature": feature,
            "n": int(len(group)),
            "r2_from_D1": float(r2_score(actual, predicted)),
            "mae_from_D1": float(np.mean(np.abs(actual - predicted))),
            "rmse_from_D1": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        })
    recon_summary = pd.DataFrame(recon_summary_rows)
    return predictions, reconstruction, recon_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/modeling/day5_branching/falsification/day5_input_audit"))
    args = parser.parse_args()

    frame = load_frame(args.path_panel, args.clusters)
    predictions, reconstruction, recon_summary = run_audit(frame)

    delta_rows = []
    for model in [column for column in predictions.columns if column.startswith("D1_plus_")]:
        delta_rows.append(proper_score_deltas(predictions, model))
    deltas = pd.DataFrame(delta_rows).sort_values(["delta_logloss", "delta_brier"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "input_audit_predictions.csv", index=False)
    reconstruction.to_csv(args.outdir / "input_reconstruction_predictions.csv", index=False)
    recon_summary.to_csv(args.outdir / "input_reconstruction_summary.csv", index=False)
    deltas.to_csv(args.outdir / "input_prediction_deltas.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps({
        "blocks": BLOCKS,
        "d1_logistic_C": 1.0,
        "reconstruction_ridge_alpha": 10.0,
        "n_predictions": int(len(predictions)),
    }, indent=2))

    print("\nD1 plus path-shape blocks ranked by log-loss delta (negative = better)")
    print(deltas.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nHeld-out reconstruction of path-shape features from D1")
    print(recon_summary.sort_values(["block", "r2_from_D1"], ascending=[True, False]).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
