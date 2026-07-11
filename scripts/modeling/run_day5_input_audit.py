#!/usr/bin/env python3
"""Classical D1 audit for predeclared non-D1 path-shape input blocks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
ASSAY_PATH = REPO / "scripts" / "modeling" / "run_day5_spatial_rydberg_assay_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_spatial_assay", ASSAY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load helpers from {ASSAY_PATH}")
assay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assay
SPEC.loader.exec_module(assay)

EPS = 1e-9


def add_path_shape_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    path = np.column_stack([
        np.zeros(len(out), dtype=float),
        *[out[f"r_d{day}"].to_numpy(float) for day in range(1, 6)],
    ])
    increments = np.diff(path, axis=1)
    second = np.diff(path, n=2, axis=1)

    endpoint = out["current_return_5d_from_branch"].to_numpy(float)
    lower = endpoint - out["distance_to_relapse_barrier"].to_numpy(float)
    upper = endpoint + out["distance_to_recovery_barrier"].to_numpy(float)
    width = np.maximum(out["barrier_width"].to_numpy(float), EPS)
    running = path[:, 1:]

    out["path_total_variation"] = np.sum(np.abs(increments), axis=1)
    out["path_efficiency"] = np.abs(endpoint) / np.maximum(out["path_total_variation"].to_numpy(float), EPS)
    out["path_curvature_l1"] = np.sum(np.abs(second), axis=1)
    out["path_reversal_count"] = np.sum(increments[:, 1:] * increments[:, :-1] < 0, axis=1)
    out["path_argmin_day"] = np.argmin(running, axis=1) + 1
    out["path_argmax_day"] = np.argmax(running, axis=1) + 1
    out["path_early_late_imbalance"] = np.sum(np.abs(increments[:, :2]), axis=1) - np.sum(np.abs(increments[:, 3:]), axis=1)

    relapse_dist = (running - lower[:, None]) / width[:, None]
    recovery_dist = (upper[:, None] - running) / width[:, None]
    out["path_min_relapse_distance_scaled"] = relapse_dist.min(axis=1)
    out["path_min_recovery_distance_scaled"] = recovery_dist.min(axis=1)
    out["path_relapse_dwell25"] = np.sum(relapse_dist <= 0.25, axis=1)
    out["path_recovery_dwell25"] = np.sum(recovery_dist <= 0.25, axis=1)
    out["path_relapse_first_day"] = np.argmin(relapse_dist, axis=1) + 1
    out["path_recovery_first_day"] = np.argmin(recovery_dist, axis=1) + 1

    for day in range(1, 5):
        out[f"trajectory_r_d{day}"] = out[f"r_d{day}"].to_numpy(float)
    return out


BLOCKS: dict[str, list[str]] = {
    "trajectory": [f"trajectory_r_d{day}" for day in range(1, 5)],
    "shape": [
        "path_total_variation",
        "path_efficiency",
        "path_curvature_l1",
        "path_reversal_count",
        "path_argmin_day",
        "path_argmax_day",
        "path_early_late_imbalance",
    ],
    "barrier_path": [
        "path_min_relapse_distance_scaled",
        "path_min_recovery_distance_scaled",
        "path_relapse_dwell25",
        "path_recovery_dwell25",
        "path_relapse_first_day",
        "path_recovery_first_day",
    ],
}
BLOCKS["all_path_shape"] = BLOCKS["trajectory"] + BLOCKS["shape"] + BLOCKS["barrier_path"]


def ridge_pipeline(alpha: float = 10.0) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def proper_score_deltas(frame: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    use = frame[["y", "D1", model, "cluster_id"]].dropna().copy()
    y = use["y"].to_numpy(int)
    p0 = np.clip(use["D1"].to_numpy(float), 1e-8, 1 - 1e-8)
    p1 = np.clip(use[model].to_numpy(float), 1e-8, 1 - 1e-8)
    ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    br0 = (p0 - y) ** 2
    br1 = (p1 - y) ** 2
    use["dll"] = ll1 - ll0
    use["dbr"] = br1 - br0
    by_cluster = use.groupby("cluster_id")[["dll", "dbr"]].mean()
    return {
        "model": model,
        "n": int(len(use)),
        "n_clusters": int(use["cluster_id"].nunique()),
        "delta_logloss": float(np.mean(ll1 - ll0)),
        "delta_brier": float(np.mean(br1 - br0)),
        "cluster_mean_delta_logloss": float(by_cluster["dll"].mean()),
        "cluster_mean_delta_brier": float(by_cluster["dbr"].mean()),
    }


def run_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = add_path_shape_features(frame)
    eligible = assay.eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))

    prediction_rows: list[dict] = []
    reconstruction_rows: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(fold_groups, start=1):
        train = frame[frame["landmark_date"] < cluster_start]
        test = frame.loc[test_group.index]
        y_train = train["y_recovery"].to_numpy(int)
        d1_train = train[assay.D1].to_numpy(float)
        d1_test = test[assay.D1].to_numpy(float)

        d1_model = assay.logistic_pipeline(1.0)
        d1_model.fit(d1_train, y_train)
        predictions: dict[str, np.ndarray] = {
            "D1": d1_model.predict_proba(d1_test)[:, 1]
        }

        for block_name, columns in BLOCKS.items():
            H_train = train[columns].to_numpy(float)
            H_test = test[columns].to_numpy(float)

            augmented = assay.logistic_pipeline(1.0)
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
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    frame = assay.load_frame(args.path_panel, args.clusters)
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
