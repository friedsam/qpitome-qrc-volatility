#!/usr/bin/env python3
"""Bounded static nonlinear probe for the day-5 branch-direction target.

This experiment follows the negative temporal ESN result. It tests whether
fixed path extrema or cheap static nonlinear maps add information beyond D1.
No temporal recurrence and no hyperparameter search are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from qpitome_qrc.baselines.logistic_offset import (
    clip_prob,
    fit_offset_logistic,
    logit,
    sigmoid,
)

MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")
RANDOM_FEATURE_DIM = 32
RANDOM_FEATURE_SEEDS = (7, 42, 123, 1001, 2026)

D1 = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "barrier_width",
]
INDEPENDENT_STATIC = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "closest_to_relapse",
    "closest_to_recovery",
]
EXTREMA = ["closest_to_relapse", "closest_to_recovery"]
MODELS = [
    "D1",
    "D1_extrema_linear",
    "D1_static_quadratic",
    "D1_random_tanh_joint",
    "D1_random_tanh_offset",
]


def add_extrema(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Claude's fixed closest-approach features from days 1..5."""
    out = frame.copy()
    running = np.column_stack(
        [out[f"r_d{day}"].to_numpy(float) for day in range(1, 6)]
    )
    endpoint = out["current_return_5d_from_branch"].to_numpy(float)
    lower = endpoint - out["distance_to_relapse_barrier"].to_numpy(float)
    upper = endpoint + out["distance_to_recovery_barrier"].to_numpy(float)
    out["closest_to_relapse"] = running.min(axis=1) - lower
    out["closest_to_recovery"] = upper - running.max(axis=1)
    return out


def load_frame(path_panel: Path, clusters_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path_panel,
        parse_dates=["branch_date", "landmark_date"],
    )
    clusters = pd.read_csv(clusters_path)[
        ["market_key", "episode_id", "cluster_id"]
    ].drop_duplicates()
    frame = frame.merge(
        clusters,
        on=["market_key", "episode_id"],
        how="left",
        validate="many_to_one",
    )
    if frame["cluster_id"].isna().any():
        raise ValueError("path-panel rows are missing cluster ids")

    starts = (
        frame.groupby("cluster_id", as_index=False)["branch_date"]
        .min()
        .rename(columns={"branch_date": "cluster_start"})
    )
    frame = frame.merge(starts, on="cluster_id", how="left", validate="many_to_one")
    frame = add_extrema(frame)

    needed = (
        D1
        + INDEPENDENT_STATIC
        + [f"r_d{day}" for day in range(1, 6)]
        + ["y_recovery", "landmark_date", "cluster_start"]
    )
    frame = (
        frame.dropna(subset=needed)
        .sort_values(["landmark_date", "market_key", "episode_id"])
        .reset_index(drop=True)
    )
    if not np.allclose(
        frame["r_d5"].to_numpy(float),
        frame["current_return_5d_from_branch"].to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("r_d5 does not match locked day-5 endpoint return")
    return frame


def logistic_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=C,
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_d1(train: pd.DataFrame) -> Pipeline:
    model = logistic_pipeline(C=1.0)
    model.fit(train[D1].to_numpy(float), train["y_recovery"].to_numpy(int))
    return model


def compute_honest_d1_predictions(frame: pd.DataFrame) -> np.ndarray:
    pred = np.full(len(frame), np.nan)
    for i, row in frame.iterrows():
        train = frame[frame["landmark_date"] < row["cluster_start"]]
        if len(train) < MIN_TRAIN or train["y_recovery"].nunique() < 2:
            continue
        pred[i] = fit_d1(train).predict_proba(
            frame.loc[[i], D1].to_numpy(float)
        )[0, 1]
    return pred


def random_tanh_features(
    train_X: np.ndarray,
    test_X: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-width random tanh map with scaling fit only on outer training rows."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    test_scaled = scaler.transform(test_X)

    rng = np.random.default_rng(seed)
    weights = rng.normal(
        0.0,
        1.0 / np.sqrt(train_scaled.shape[1]),
        size=(train_scaled.shape[1], RANDOM_FEATURE_DIM),
    )
    bias = rng.uniform(-1.0, 1.0, size=RANDOM_FEATURE_DIM)
    return (
        np.tanh(train_scaled @ weights + bias),
        np.tanh(test_scaled @ weights + bias),
    )


def evaluate(frame: pd.DataFrame, honest_d1: np.ndarray) -> pd.DataFrame:
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
        test = frame.iloc[[i]]

        d1_model = fit_d1(train)
        p_d1 = float(d1_model.predict_proba(test[D1].to_numpy(float))[0, 1])

        extrema_model = logistic_pipeline(C=1.0)
        extrema_cols = D1 + EXTREMA
        extrema_model.fit(train[extrema_cols].to_numpy(float), y_train)
        p_extrema = float(
            extrema_model.predict_proba(test[extrema_cols].to_numpy(float))[0, 1]
        )

        quadratic = Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                (
                    "logit",
                    LogisticRegression(
                        C=0.1,
                        max_iter=5000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        quadratic.fit(train[INDEPENDENT_STATIC].to_numpy(float), y_train)
        p_quadratic = float(
            quadratic.predict_proba(test[INDEPENDENT_STATIC].to_numpy(float))[0, 1]
        )

        joint_predictions = []
        offset_predictions = []
        for seed in RANDOM_FEATURE_SEEDS:
            features_train, features_test = random_tanh_features(
                train[INDEPENDENT_STATIC].to_numpy(float),
                test[INDEPENDENT_STATIC].to_numpy(float),
                seed,
            )

            joint_train = np.column_stack(
                [train[D1].to_numpy(float), features_train]
            )
            joint_test = np.column_stack(
                [test[D1].to_numpy(float), features_test]
            )
            joint = logistic_pipeline(C=0.1)
            joint.fit(joint_train, y_train)
            joint_predictions.append(float(joint.predict_proba(joint_test)[0, 1]))

            valid = np.isfinite(honest_d1[train_idx])
            if valid.sum() >= MIN_TRAIN and np.unique(y_train[valid]).size == 2:
                feature_scaler = StandardScaler()
                correction_train = feature_scaler.fit_transform(
                    features_train[valid]
                )
                correction_test = feature_scaler.transform(features_test)
                beta, intercept = fit_offset_logistic(
                    correction_train,
                    y_train[valid],
                    logit(honest_d1[train_idx][valid]),
                    l2=10.0,
                )
                eta = (
                    float(logit(p_d1))
                    + intercept
                    + (correction_test @ beta).item()
                )
                offset_predictions.append(float(sigmoid(eta)))

        rows.append(
            {
                "row_id": int(i),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "y": int(row["y_recovery"]),
                "D1": p_d1,
                "D1_extrema_linear": p_extrema,
                "D1_static_quadratic": p_quadratic,
                "D1_random_tanh_joint": float(np.mean(joint_predictions)),
                "D1_random_tanh_offset": (
                    float(np.mean(offset_predictions))
                    if offset_predictions
                    else np.nan
                ),
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
        "pr_auc": (
            float(average_precision_score(y, p))
            if np.unique(y).size == 2
            else np.nan
        ),
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
        "leave_nikkei_out_eval": predictions[
            predictions["market_key"] != "nikkei_225"
        ],
        "nikkei_only_eval": predictions[
            predictions["market_key"] == "nikkei_225"
        ],
    }
    summary_rows = []
    paired_rows = []
    cluster_rows = []
    for group_name, group in groups.items():
        for model in MODELS:
            row = score(group, model)
            row["group"] = group_name
            summary_rows.append(row)

            cluster = cluster_metrics(group, model)
            cluster["group"] = group_name
            cluster_rows.append(cluster)

            if model != "D1":
                paired = paired_delta(group, model)
                paired["group"] = group_name
                paired_rows.append(paired)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(cluster_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-panel",
        type=Path,
        default=Path("scratch/path_panel_day0_5.csv"),
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path(
            "results/diagnostics/cross_market_crisis_clusters_v2/"
            "branch_sync_cluster_detail.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/tmp/qpitome_branch_static_probe"),
    )
    args = parser.parse_args()

    frame = load_frame(args.path_panel, args.clusters)
    honest_d1 = compute_honest_d1_predictions(frame)
    predictions = evaluate(frame, honest_d1)
    summary, paired, cluster = summarize(predictions)

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    summary.to_csv(args.outdir / "summary_metrics.csv", index=False)
    paired.to_csv(args.outdir / "paired_score_deltas.csv", index=False)
    cluster.to_csv(args.outdir / "cluster_weighted_metrics.csv", index=False)
    manifest = {
        "purpose": "Bounded static nonlinear probe after negative temporal ESN",
        "models": MODELS,
        "independent_static_inputs": INDEPENDENT_STATIC,
        "random_feature_dimension": RANDOM_FEATURE_DIM,
        "random_feature_seeds": RANDOM_FEATURE_SEEDS,
        "status": "exploratory; no hyperparameter search",
    }
    (args.outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=list)
    )

    print("\nSummary metrics")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPaired deltas versus D1 (negative = better)")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCluster-weighted metrics")
    print(cluster.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved exploratory outputs to {args.outdir}")


if __name__ == "__main__":
    main()
