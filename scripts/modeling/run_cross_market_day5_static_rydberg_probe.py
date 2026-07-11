#!/usr/bin/env python3
"""State-conditioned memoryless Rydberg probe for the day-5 branch target."""

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
from qpitome_qrc.qrc.rydberg_reservoir import RydbergQRCConfig, build_rydberg_feature_matrix

MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")
MASK_COUNT = 4
MASK_SEED = 20260711
MASK_EVOLUTION_US = 0.55

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
    if not np.allclose(
        frame["r_d5"].to_numpy(float),
        frame["current_return_5d_from_branch"].to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("r_d5 does not match locked day-5 endpoint return")
    return frame


def logistic_pipeline(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
        ]
    )


def make_masks() -> np.ndarray:
    rng = np.random.default_rng(MASK_SEED)
    return rng.normal(0.0, 1.0 / np.sqrt(len(STATIC)), size=(MASK_COUNT, 2, len(STATIC)))


def make_mask_windows(
    train_X: np.ndarray,
    all_X: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_X)
    scaled = scaler.transform(all_X)
    projected = np.einsum("sd,mcd->smc", scaled, masks)
    return np.tanh(projected)


def rydberg_config() -> RydbergQRCConfig:
    return RydbergQRCConfig(
        geometry="dual_chain",
        n_atoms_slow=4,
        n_atoms_fast=4,
        spacing_slow_um=9.0,
        spacing_fast_um=15.0,
        row_gap_um=14.0,
        lookback_days=MASK_COUNT,
        anchor_count=MASK_COUNT,
        anchor_policy="even",
        total_time_us=MASK_COUNT * MASK_EVOLUTION_US,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_frac=0.5,
        omega_mode="encode_rate",
        encoding="plateau",
        memory_mode="memoryless",
        observable_mode="n",
        collect_anchor_features=True,
        shots=None,
    )


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    masks = make_masks()
    config = rydberg_config()
    rows = []

    for i, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train_mask = frame["landmark_date"] < row["cluster_start"]
        train_idx = np.flatnonzero(train_mask.to_numpy())
        if len(train_idx) < MIN_TRAIN:
            continue
        train = frame.iloc[train_idx]
        if train["y_recovery"].nunique() < 2:
            continue

        y_train = train["y_recovery"].to_numpy(int)
        d1_model = logistic_pipeline(C=1.0)
        d1_model.fit(train[D1].to_numpy(float), y_train)
        p_d1 = float(d1_model.predict_proba(frame.loc[[i], D1].to_numpy(float))[0, 1])

        windows = make_mask_windows(
            train[STATIC].to_numpy(float),
            frame[STATIC].to_numpy(float),
            masks,
        )
        features = build_rydberg_feature_matrix(windows, config)
        joint_train = np.column_stack([train[D1].to_numpy(float), features[train_idx]])
        joint_test = np.column_stack([frame.loc[[i], D1].to_numpy(float), features[[i]]])

        joint = logistic_pipeline(C=0.1)
        joint.fit(joint_train, y_train)
        p_joint = float(joint.predict_proba(joint_test)[0, 1])

        rows.append(
            {
                "row_id": int(i),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "y": int(row["y_recovery"]),
                "D1": p_d1,
                "D1_plus_rydberg_joint": p_joint,
            }
        )

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


def paired_delta(group: pd.DataFrame, model: str) -> dict:
    use = group[["y", "D1", model, "cluster_id"]].dropna().copy()
    y = use["y"].to_numpy(int)
    p_d1 = clip_prob(use["D1"].to_numpy(float))
    p_model = clip_prob(use[model].to_numpy(float))
    ll_d1 = -(y * np.log(p_d1) + (1 - y) * np.log(1 - p_d1))
    ll_model = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
    br_d1 = (p_d1 - y) ** 2
    br_model = (p_model - y) ** 2
    use["dll"] = ll_model - ll_d1
    use["dbr"] = br_model - br_d1
    cluster = use.groupby("cluster_id")[["dll", "dbr"]].mean()
    return {
        "model": model,
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
    rows = []
    for cluster_id, subset in group.dropna(subset=[model]).groupby("cluster_id"):
        y = subset["y"].to_numpy(int)
        p = clip_prob(subset[model].to_numpy(float))
        rows.append(
            {
                "cluster_id": cluster_id,
                "n": len(subset),
                "logloss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(np.mean((p - y) ** 2)),
            }
        )
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
    summary_rows = []
    paired_rows = []
    cluster_rows = []
    for group_name, group in groups.items():
        for model in ("D1", "D1_plus_rydberg_joint"):
            row = score(group, model)
            row["group"] = group_name
            summary_rows.append(row)
            crow = cluster_metrics(group, model)
            crow["group"] = group_name
            cluster_rows.append(crow)
        prow = paired_delta(group, "D1_plus_rydberg_joint")
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
    parser.add_argument("--outdir", type=Path, default=Path("/tmp/qpitome_branch_rydberg_probe"))
    args = parser.parse_args()

    frame = load_frame(args.path_panel, args.clusters)
    predictions = evaluate(frame)
    summary, paired, cluster = summarize(predictions)

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    summary.to_csv(args.outdir / "summary_metrics.csv", index=False)
    paired.to_csv(args.outdir / "paired_score_deltas.csv", index=False)
    cluster.to_csv(args.outdir / "cluster_weighted_metrics.csv", index=False)
    manifest = {
        "purpose": "Bounded state-conditioned memoryless Rydberg probe",
        "static_inputs": STATIC,
        "mask_count": MASK_COUNT,
        "mask_seed": MASK_SEED,
        "mask_evolution_us": MASK_EVOLUTION_US,
        "feature_dimension": MASK_COUNT * 8,
        "rydberg_config": rydberg_config().__dict__,
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
