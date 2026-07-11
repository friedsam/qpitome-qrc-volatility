#!/usr/bin/env python3
"""Protected classical residual test for the least D1-reconstructible path features."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


input_audit = load_module(
    "day5_input_audit",
    REPO / "scripts" / "modeling" / "run_day5_input_audit.py",
)
crossfit = load_module(
    "day5_residualized_crossfit",
    REPO / "scripts" / "modeling" / "run_day5_residualized_rydberg_crossfit_shard.py",
)
base = crossfit.base

FEATURE_BLOCKS: dict[str, list[str]] = {
    "path_efficiency": ["path_efficiency"],
    "path_reversal_count": ["path_reversal_count"],
    "path_early_late_imbalance": ["path_early_late_imbalance"],
    "trajectory_r_d1": ["trajectory_r_d1"],
    "compact_four": [
        "path_efficiency",
        "path_reversal_count",
        "path_early_late_imbalance",
        "trajectory_r_d1",
    ],
}
RIDGE_ALPHA = 10.0
OFFSET_L2 = 100.0
RNG_SEED = 20260711


def score_deltas(predictions: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    use = predictions[["y", "D1", model, "cluster_id"]].dropna().copy()
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


def run(frame: pd.DataFrame) -> pd.DataFrame:
    frame = input_audit.add_path_shape_features(frame)
    eligible = base.assay.eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))
    records: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(fold_groups, start=1):
        train = frame[frame["landmark_date"] < cluster_start].copy()
        test = frame.loc[test_group.index].copy()
        y_train = train["y_recovery"].to_numpy(int)
        d1_train = train[base.assay.D1].to_numpy(float)
        d1_test = test[base.assay.D1].to_numpy(float)

        X_train = base.d1_basis(d1_train, "quadratic")
        X_test = base.d1_basis(d1_test, "quadratic")
        cf_positions, cf_logits = crossfit.historical_crossfit_d1_logits(train)
        if len(cf_positions) < crossfit.MIN_CORRECTION_TRAIN or np.unique(y_train[cf_positions]).size < 2:
            raise RuntimeError(
                f"Insufficient cross-fitted rows for cluster {cluster_start}: {len(cf_positions)}"
            )

        d1_model = base.assay.logistic_pipeline(1.0)
        d1_model.fit(d1_train, y_train)
        p_test_d1 = d1_model.predict_proba(d1_test)[:, 1]
        offset_test = base.assay.logit(p_test_d1)
        predictions: dict[str, np.ndarray] = {"D1": p_test_d1}

        rng = np.random.default_rng(RNG_SEED + int(pd.Timestamp(cluster_start).value % 2**31))
        for block_name, columns in FEATURE_BLOCKS.items():
            H_train = train[columns].to_numpy(float)
            H_test = test[columns].to_numpy(float)
            R_train, R_test = base.residualize_train_test(
                X_train, H_train, X_test, H_test, RIDGE_ALPHA
            )

            corrected = []
            gaussian = []
            G_train = rng.normal(size=H_train.shape)
            G_test = rng.normal(size=H_test.shape)
            for row_number in range(len(test)):
                corrected.append(base.assay.fit_offset_predict(
                    R_train[cf_positions],
                    y_train[cf_positions],
                    R_test[[row_number]],
                    cf_logits,
                    float(offset_test[row_number]),
                    OFFSET_L2,
                ))
                gaussian.append(base.assay.fit_offset_predict(
                    G_train[cf_positions],
                    y_train[cf_positions],
                    G_test[[row_number]],
                    cf_logits,
                    float(offset_test[row_number]),
                    OFFSET_L2,
                ))
            predictions[f"protected_{block_name}"] = np.asarray(corrected)
            predictions[f"gaussian_{block_name}"] = np.asarray(gaussian)

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
                "n_correction_train": int(len(cf_positions)),
            }
            for name, values in predictions.items():
                record[name] = float(values[local_row])
            records.append(record)

        print(
            f"fold {fold_number}/{len(fold_groups)} cluster_start={pd.Timestamp(cluster_start).date()} "
            f"train={len(train)} correction_train={len(cf_positions)} test={len(test)}",
            flush=True,
        )

    return pd.DataFrame(records).sort_values("row_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    frame = base.assay.load_frame(args.path_panel, args.clusters)
    predictions = run(frame)
    models = [
        column for column in predictions.columns
        if column.startswith("protected_") or column.startswith("gaussian_")
    ]
    summary = pd.DataFrame([score_deltas(predictions, model) for model in models]).sort_values(
        ["delta_logloss", "delta_brier"]
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "protected_input_predictions.csv", index=False)
    summary.to_csv(args.outdir / "protected_input_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps({
        "feature_blocks": FEATURE_BLOCKS,
        "residualizer": "quadratic_D1",
        "ridge_alpha": RIDGE_ALPHA,
        "offset_l2": OFFSET_L2,
        "inner_crossfit_min_train": crossfit.INNER_CROSSFIT_MIN_TRAIN,
        "min_correction_train": crossfit.MIN_CORRECTION_TRAIN,
        "n_predictions": int(len(predictions)),
    }, indent=2))

    print("\nProtected path features ranked by log-loss delta (negative = better)")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
