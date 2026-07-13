#!/usr/bin/env python3
"""Local-detuning Rydberg probe for the day-5 branch-direction target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.logistic_offset import clip_prob
from qpitome_qrc.day5.protocol import (
    D1,
    EVAL_START,
    MIN_TRAIN,
    STATIC,
    add_extrema,
    load_frame,
)
from qpitome_qrc.evaluation.binary import logistic_pipeline
from qpitome_qrc.evaluation.scoring import binary_summary, cluster_weighted_summary
from qpitome_qrc.qrc.local_detuning_reservoir import (
    LocalDetuningConfig,
    build_local_detuning_feature_matrix,
)

MODEL = "D1_plus_local_rydberg_joint"


def static_to_local_patterns(train_X: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Map five train-standardized coordinates to eight site coefficients."""
    scaler = StandardScaler().fit(train_X)
    z = scaler.transform(X)
    site_values = np.column_stack(
        [
            z[:, 0],
            z[:, 1],
            z[:, 2],
            z[:, 3],
            z[:, 4],
            (z[:, 0] - z[:, 1]) / np.sqrt(2.0),
            (z[:, 3] - z[:, 4]) / np.sqrt(2.0),
            (z[:, 1] - z[:, 2] + z[:, 3] - z[:, 4]) / 2.0,
        ]
    )
    return 0.5 * (np.tanh(site_values) + 1.0)


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    config = LocalDetuningConfig()
    rows = []

    for i, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train_idx = np.flatnonzero((frame["landmark_date"] < row["cluster_start"]).to_numpy())
        if len(train_idx) < MIN_TRAIN:
            continue
        train = frame.iloc[train_idx]
        if train["y_recovery"].nunique() < 2:
            continue

        y_train = train["y_recovery"].to_numpy(int)
        d1_model = logistic_pipeline(C=1.0)
        d1_model.fit(train[D1].to_numpy(float), y_train)
        p_d1 = float(d1_model.predict_proba(frame.loc[[i], D1].to_numpy(float))[0, 1])

        fold_rows = np.concatenate([train_idx, np.array([i], dtype=int)])
        patterns = static_to_local_patterns(
            train[STATIC].to_numpy(float),
            frame.iloc[fold_rows][STATIC].to_numpy(float),
        )
        rydberg_features = build_local_detuning_feature_matrix(patterns, config)
        H_train = rydberg_features[:-1]
        H_test = rydberg_features[-1:]

        joint = logistic_pipeline(C=0.1)
        joint.fit(np.column_stack([train[D1].to_numpy(float), H_train]), y_train)
        p_joint = float(
            joint.predict_proba(
                np.column_stack([frame.loc[[i], D1].to_numpy(float), H_test])
            )[0, 1]
        )

        rows.append(
            {
                "row_id": int(i),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "y": int(row["y_recovery"]),
                "D1": p_d1,
                MODEL: p_joint,
            }
        )

    return pd.DataFrame(rows)


def score(group: pd.DataFrame, model: str) -> dict:
    """Return the historical probe summary schema via shared scoring."""

    return binary_summary(group, model, clip=1e-6)


def paired_delta(group: pd.DataFrame) -> dict:
    use = group[["y", "D1", MODEL, "cluster_id"]].dropna().copy()
    y = use["y"].to_numpy(int)
    p_d1 = clip_prob(use["D1"].to_numpy(float))
    p_model = clip_prob(use[MODEL].to_numpy(float))
    ll_d1 = -(y * np.log(p_d1) + (1 - y) * np.log(1 - p_d1))
    ll_model = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
    br_d1 = (p_d1 - y) ** 2
    br_model = (p_model - y) ** 2
    use["dll"] = ll_model - ll_d1
    use["dbr"] = br_model - br_d1
    cluster = use.groupby("cluster_id")[["dll", "dbr"]].mean()
    return {
        "model": MODEL,
        "n": int(len(use)),
        "n_clusters": int(use["cluster_id"].nunique()),
        "D1_logloss_matched": float(ll_d1.mean()),
        "model_logloss_matched": float(ll_model.mean()),
        "delta_logloss": float((ll_model - ll_d1).mean()),
        "D1_brier_matched": float(br_d1.mean()),
        "model_brier_matched": float(br_model.mean()),
        "delta_brier": float((br_model - br_d1).mean()),
        "cluster_mean_delta_logloss": float(cluster["dll"].mean()),
        "cluster_mean_delta_brier": float(cluster["dbr"].mean()),
    }


def cluster_metrics(group: pd.DataFrame, model: str) -> dict:
    """Return the historical cluster-weighted schema via shared scoring."""

    return cluster_weighted_summary(group, model, clip=1e-6)


def summarize(predictions: pd.DataFrame):
    groups = {
        "all_post1990_purged": predictions,
        "validation_non_spy": predictions[predictions["market_key"] != "spy"],
        "leave_nikkei_out_eval": predictions[predictions["market_key"] != "nikkei_225"],
        "nikkei_only_eval": predictions[predictions["market_key"] == "nikkei_225"],
    }
    summary_rows = []
    paired_rows = []
    cluster_rows = []
    for group_name, group in groups.items():
        for model in ("D1", MODEL):
            row = score(group, model)
            row["group"] = group_name
            summary_rows.append(row)
            crow = cluster_metrics(group, model)
            crow["group"] = group_name
            cluster_rows.append(crow)
        prow = paired_delta(group)
        prow["group"] = group_name
        paired_rows.append(prow)
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows), pd.DataFrame(cluster_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/modeling/day5_branching/static_rydberg/static_rydberg_probe"))
    args = parser.parse_args()

    frame = load_frame(args.path_panel, args.clusters)
    predictions = evaluate(frame)
    summary, paired, cluster = summarize(predictions)

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    summary.to_csv(args.outdir / "summary_metrics.csv", index=False)
    paired.to_csv(args.outdir / "paired_score_deltas.csv", index=False)
    cluster.to_csv(args.outdir / "cluster_weighted_metrics.csv", index=False)
    config = LocalDetuningConfig()
    manifest = {
        "purpose": "Bounded native local-detuning Rydberg probe",
        "static_inputs": STATIC,
        "site_encoding": "five direct coordinates plus three fixed contrasts",
        "feature_dimension": 36,
        "local_detuning_config": {
            "evolution_time_us": config.evolution_time_us,
            "global_omega_rad_us": config.global_omega_rad_us,
            "global_delta_rad_us": config.global_delta_rad_us,
            "local_delta_rad_us": config.local_delta_rad_us,
            "reservoir": config.reservoir.__dict__,
        },
        "status": "exploratory exact-state simulation; no parameter sweep",
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\nSummary metrics")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPaired deltas versus D1 (negative = better)")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCluster-weighted metrics")
    print(cluster.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved exploratory outputs to {args.outdir}")


if __name__ == "__main__":
    main()
