#!/usr/bin/env python3
"""Frozen outer test for differential-pair local-detuning Rydberg features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.logistic_offset import clip_prob
from qpitome_qrc.day5.protocol import (
    D1,
    EVAL_START,
    MIN_TRAIN,
    STATIC,
    add_extrema,
    differential_patterns,
    load_frame,
    rydberg_config,
)
from qpitome_qrc.evaluation.binary import logistic_pipeline
from qpitome_qrc.evaluation.scoring import binary_summary, cluster_weighted_summary
from qpitome_qrc.qrc.local_detuning_reservoir import build_local_detuning_feature_matrix

MODEL = "D1_plus_differential_local_rydberg_joint"


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    config = rydberg_config()
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

        d1 = logistic_pipeline(1.0)
        d1.fit(train[D1].to_numpy(float), y_train)
        p_d1 = float(d1.predict_proba(frame.loc[[i], D1].to_numpy(float))[0, 1])

        fold_idx = np.concatenate([train_idx, np.array([i], dtype=int)])
        patterns = differential_patterns(
            train[STATIC].to_numpy(float),
            frame.iloc[fold_idx][STATIC].to_numpy(float),
        )
        features = build_local_detuning_feature_matrix(patterns, config)
        H_train, H_test = features[:-1], features[-1:]

        joint = logistic_pipeline(0.1)
        joint.fit(np.column_stack([train[D1].to_numpy(float), H_train]), y_train)
        p_joint = float(
            joint.predict_proba(
                np.column_stack([frame.loc[[i], D1].to_numpy(float), H_test])
            )[0, 1]
        )

        rows.append({
            "row_id": int(i),
            "market_key": row["market_key"],
            "episode_id": int(row["episode_id"]),
            "cluster_id": row["cluster_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
            "D1": p_d1,
            MODEL: p_joint,
        })
    return pd.DataFrame(rows)


def score(group: pd.DataFrame, model: str) -> dict:
    """Return the historical probe summary schema via shared scoring."""

    return binary_summary(group, model, clip=1e-6)


def paired_delta(group: pd.DataFrame) -> dict:
    use = group[["y", "D1", MODEL, "cluster_id"]].dropna().copy()
    y = use["y"].to_numpy(int)
    p0 = clip_prob(use["D1"].to_numpy(float))
    p1 = clip_prob(use[MODEL].to_numpy(float))
    ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    br0 = (p0 - y) ** 2
    br1 = (p1 - y) ** 2
    use["dll"] = ll1 - ll0
    use["dbr"] = br1 - br0
    cluster = use.groupby("cluster_id")[["dll", "dbr"]].mean()
    return {
        "model": MODEL,
        "n": int(len(use)),
        "n_clusters": int(use["cluster_id"].nunique()),
        "D1_logloss_matched": float(ll0.mean()),
        "model_logloss_matched": float(ll1.mean()),
        "delta_logloss": float((ll1 - ll0).mean()),
        "D1_brier_matched": float(br0.mean()),
        "model_brier_matched": float(br1.mean()),
        "delta_brier": float((br1 - br0).mean()),
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
    summary_rows, paired_rows, cluster_rows = [], [], []
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
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("/tmp/qpitome_branch_differential_rydberg_probe"))
    args = parser.parse_args()

    frame = load_frame(args.path_panel, args.clusters)
    predictions = evaluate(frame)
    summary, paired, cluster = summarize(predictions)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    summary.to_csv(args.outdir / "summary_metrics.csv", index=False)
    paired.to_csv(args.outdir / "paired_score_deltas.csv", index=False)
    cluster.to_csv(args.outdir / "cluster_weighted_metrics.csv", index=False)
    cfg = rydberg_config()
    manifest = {
        "purpose": "Frozen outer test of differential-pair local-detuning Rydberg features",
        "static_inputs": STATIC,
        "feature_dimension": 55,
        "encoding": "five complementary atom pairs",
        "readout_C": 0.1,
        "local_detuning_config": {
            "evolution_time_us": cfg.evolution_time_us,
            "global_omega_rad_us": cfg.global_omega_rad_us,
            "global_delta_rad_us": cfg.global_delta_rad_us,
            "local_delta_rad_us": cfg.local_delta_rad_us,
            "reservoir": cfg.reservoir.__dict__,
        },
        "status": "frozen outer evaluation after successful synthetic diagnostic",
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("\nSummary metrics")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPaired deltas versus D1 (negative = better)")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCluster-weighted metrics")
    print(cluster.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to {args.outdir}")


if __name__ == "__main__":
    main()
