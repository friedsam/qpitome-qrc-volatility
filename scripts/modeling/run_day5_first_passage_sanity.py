#!/usr/bin/env python3
"""Minimal sanity test for construction-implied Day-5 first-passage predictability.

The test asks whether the zero-parameter corridor-position probability

    p_upper = distance_to_relapse / barrier_width

already explains the locked Day-5 recovery label nearly as well as the fitted
four-column D1 logistic model.  It also verifies the exact feature identities
and reports the effective numerical dimension of D1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
EPS = 1e-8


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protected = load_module(
    "day5_protected_input",
    REPO / "scripts" / "modeling" / "run_day5_protected_input_residual.py",
)
base = protected.base


def corridor_position(frame: pd.DataFrame) -> np.ndarray:
    width = frame["barrier_width"].to_numpy(float)
    if np.any(width <= 0):
        raise ValueError("barrier_width must be positive")
    return frame["distance_to_relapse_barrier"].to_numpy(float) / width


def metric_row(name: str, y: np.ndarray, p: np.ndarray, cluster_id: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=int)
    per_row = pd.DataFrame({
        "cluster_id": cluster_id,
        "logloss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (p - y) ** 2,
    })
    by_cluster = per_row.groupby("cluster_id")[["logloss", "brier"]].mean()
    return {
        "model": name,
        "n": int(len(y)),
        "n_clusters": int(pd.Series(cluster_id).nunique()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "cluster_mean_logloss": float(by_cluster["logloss"].mean()),
        "cluster_mean_brier": float(by_cluster["brier"].mean()),
    }


def run(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eligible = base.assay.eligible_rows(frame)
    eval_frame = frame.loc[eligible].copy()
    records: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(
        eval_frame.groupby("cluster_start", sort=True), start=1
    ):
        train = frame[frame["landmark_date"] < cluster_start].copy()
        test = frame.loc[test_group.index].copy()
        y_train = train["y_recovery"].to_numpy(int)
        if len(train) == 0 or np.unique(y_train).size < 2:
            continue

        p_null = np.clip(corridor_position(test), EPS, 1.0 - EPS)
        p_prior = np.full(len(test), np.clip(y_train.mean(), EPS, 1.0 - EPS))

        lambda_model = base.assay.logistic_pipeline(1.0)
        lambda_model.fit(corridor_position(train).reshape(-1, 1), y_train)
        p_lambda_cal = lambda_model.predict_proba(corridor_position(test).reshape(-1, 1))[:, 1]

        d1_model = base.assay.logistic_pipeline(1.0)
        d1_model.fit(train[base.assay.D1].to_numpy(float), y_train)
        p_d1 = d1_model.predict_proba(test[base.assay.D1].to_numpy(float))[:, 1]

        for local_row, row_index in enumerate(test.index):
            row = test.loc[row_index]
            records.append({
                "row_id": int(row_index),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "cluster_start": cluster_start,
                "y": int(row["y_recovery"]),
                "prior": float(p_prior[local_row]),
                "first_passage_null": float(p_null[local_row]),
                "calibrated_lambda": float(p_lambda_cal[local_row]),
                "D1": float(p_d1[local_row]),
            })

        print(
            f"fold {fold_number} cluster_start={pd.Timestamp(cluster_start).date()} "
            f"train={len(train)} test={len(test)}",
            flush=True,
        )

    predictions = pd.DataFrame(records).sort_values("row_id").reset_index(drop=True)
    y = predictions["y"].to_numpy(int)
    cluster_id = predictions["cluster_id"].to_numpy()
    summary = pd.DataFrame([
        metric_row(name, y, predictions[name].to_numpy(float), cluster_id)
        for name in ["prior", "first_passage_null", "calibrated_lambda", "D1"]
    ]).sort_values("logloss").reset_index(drop=True)

    d1 = frame.loc[eligible, base.assay.D1].to_numpy(float)
    z = StandardScaler().fit_transform(d1)
    singular_values = np.linalg.svd(z, full_matrices=False, compute_uv=False)
    variance = singular_values**2
    explained = variance / variance.sum()

    width_error = (
        frame.loc[eligible, "barrier_width"].to_numpy(float)
        - frame.loc[eligible, "distance_to_recovery_barrier"].to_numpy(float)
        - frame.loc[eligible, "distance_to_relapse_barrier"].to_numpy(float)
    )
    lam = corridor_position(frame.loc[eligible])
    diagnostics = {
        "n_eligible": int(eligible.sum()),
        "max_abs_width_identity_error": float(np.max(np.abs(width_error))),
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "d1_singular_values": singular_values.tolist(),
        "d1_explained_variance_ratio": explained.tolist(),
        "d1_cumulative_explained_variance_ratio": np.cumsum(explained).tolist(),
        "prediction_correlation_null_vs_d1": float(
            np.corrcoef(predictions["first_passage_null"], predictions["D1"])[0, 1]
        ),
        "mean_abs_prediction_difference_null_vs_d1": float(
            np.mean(np.abs(predictions["first_passage_null"] - predictions["D1"]))
        ),
        "delta_logloss_d1_minus_null": float(
            summary.set_index("model").loc["D1", "logloss"]
            - summary.set_index("model").loc["first_passage_null", "logloss"]
        ),
        "delta_brier_d1_minus_null": float(
            summary.set_index("model").loc["D1", "brier"]
            - summary.set_index("model").loc["first_passage_null", "brier"]
        ),
    }
    return predictions, summary, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    frame = base.assay.load_frame(args.path_panel, args.clusters)
    predictions, summary, diagnostics = run(frame)

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    (args.outdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    print("\nDay-5 first-passage sanity test")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDiagnostics")
    print(json.dumps(diagnostics, indent=2))
    print(
        "\nInterpretation: if the zero-parameter first_passage_null nearly matches D1, "
        "most D1 performance is construction-implied corridor geometry rather than "
        "additional learned market structure."
    )


if __name__ == "__main__":
    main()
