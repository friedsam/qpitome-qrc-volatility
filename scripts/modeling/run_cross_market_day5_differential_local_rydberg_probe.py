#!/usr/bin/env python3
"""Frozen outer test for differential-pair local-detuning Rydberg features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.logistic_offset import clip_prob
from qpitome_qrc.qrc.local_detuning_reservoir import LocalDetuningConfig, build_local_detuning_feature_matrix
from qpitome_qrc.qrc.rydberg_reservoir import RydbergQRCConfig

MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")
MODEL = "D1_plus_differential_local_rydberg_joint"
D1 = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "barrier_width",
]
STATIC = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "closest_to_relapse",
    "closest_to_recovery",
]


def add_extrema(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    running = np.column_stack([out[f"r_d{day}"].to_numpy(float) for day in range(1, 6)])
    endpoint = out["current_return_5d_from_branch"].to_numpy(float)
    lower = endpoint - out["distance_to_relapse_barrier"].to_numpy(float)
    upper = endpoint + out["distance_to_recovery_barrier"].to_numpy(float)
    out["closest_to_relapse"] = running.min(axis=1) - lower
    out["closest_to_recovery"] = upper - running.max(axis=1)
    return out


def load_frame(path_panel: Path, clusters_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path_panel, parse_dates=["branch_date", "landmark_date"])
    clusters = pd.read_csv(clusters_path)[["market_key", "episode_id", "cluster_id"]].drop_duplicates()
    frame = frame.merge(clusters, on=["market_key", "episode_id"], how="left", validate="many_to_one")
    if frame["cluster_id"].isna().any():
        raise ValueError("path-panel rows are missing cluster ids")
    starts = (
        frame.groupby("cluster_id", as_index=False)["branch_date"]
        .min()
        .rename(columns={"branch_date": "cluster_start"})
    )
    frame = frame.merge(starts, on="cluster_id", how="left", validate="many_to_one")
    frame = add_extrema(frame)
    needed = D1 + STATIC + [f"r_d{day}" for day in range(1, 6)] + ["y_recovery", "landmark_date", "cluster_start"]
    frame = frame.dropna(subset=needed).sort_values(["landmark_date", "market_key", "episode_id"]).reset_index(drop=True)
    if not np.allclose(frame["r_d5"].to_numpy(float), frame["current_return_5d_from_branch"].to_numpy(float), atol=1e-12, rtol=0):
        raise ValueError("r_d5 does not match locked day-5 endpoint return")
    return frame


def logistic_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
    ])


def differential_patterns(train_X: np.ndarray, X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(train_X)
    z = scaler.transform(X)
    positive = 0.5 * (1.0 + np.tanh(z))
    negative = 1.0 - positive
    patterns = np.empty((len(z), 10), dtype=float)
    patterns[:, 0::2] = positive
    patterns[:, 1::2] = negative
    return patterns


def rydberg_config() -> LocalDetuningConfig:
    reservoir = RydbergQRCConfig(
        geometry="chain",
        chain_atoms=10,
        chain_spacing_um=7.5,
        observable_mode="n_nn",
        collect_anchor_features=False,
        memory_mode="memoryless",
        shots=None,
    )
    return LocalDetuningConfig(
        reservoir=reservoir,
        evolution_time_us=0.55,
        global_omega_rad_us=6.0,
        global_delta_rad_us=6.0,
        local_delta_rad_us=4.0,
    )


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
        p_joint = float(joint.predict_proba(np.column_stack([frame.loc[[i], D1].to_numpy(float), H_test]))[0, 1])

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
    use = group[["y", model]].dropna()
    y = use["y"].to_numpy(int)
    p = clip_prob(use[model].to_numpy(float))
    return {
        "model": model,
        "n": int(len(use)),
        "recovery_rate": float(y.mean()) if len(y) else np.nan,
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "logloss": float(log_loss(y, p)) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }


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
    rows = []
    for cluster_id, subset in group.dropna(subset=[model]).groupby("cluster_id"):
        y = subset["y"].to_numpy(int)
        p = clip_prob(subset[model].to_numpy(float))
        rows.append({
            "cluster_id": cluster_id,
            "n": len(subset),
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(np.mean((p - y) ** 2)),
        })
    detail = pd.DataFrame(rows)
    return {
        "model": model,
        "n_clusters": int(len(detail)),
        "cluster_mean_logloss": float(detail["logloss"].mean()),
        "cluster_mean_brier": float(detail["brier"].mean()),
        "median_cluster_size": float(detail["n"].median()),
    }


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
            row = score(group, model); row["group"] = group_name; summary_rows.append(row)
            crow = cluster_metrics(group, model); crow["group"] = group_name; cluster_rows.append(crow)
        prow = paired_delta(group); prow["group"] = group_name; paired_rows.append(prow)
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
