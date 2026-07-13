#!/usr/bin/env python3
"""Structural audit for the provisional Task B destination target.

Reads the episode table created by ``audit_branch_destination_targets.py`` and
checks origin-regime composition, temporal concentration, class drift, and
whether simple causal features remain useful within each origin regime.

The target remains 8-week observed cumulative return with a +/-1% neutral zone
unless overridden. This script does not fit a multivariate classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "return_1w",
    "momentum_4w",
    "momentum_13w",
    "vol_4w",
    "vol_13w",
    "bull_probability",
    "long_regime_uncertainty",
    "one_week_probability_motion",
]
EPS = 1e-12


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_loss(y: np.ndarray, probability: np.ndarray | float) -> float:
    p = np.asarray(probability, dtype=float)
    if p.ndim == 0:
        p = np.full(len(y), float(p))
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, probability: np.ndarray | float) -> float:
    p = np.asarray(probability, dtype=float)
    if p.ndim == 0:
        p = np.full(len(y), float(p))
    return float(np.mean((y - p) ** 2))


def prepare(path: Path, horizon: int, neutral_zone: float, split_date: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    target = f"future_return_{horizon}w"
    if target not in frame:
        raise ValueError(f"Missing target column {target!r}")
    frame = frame[frame[target].notna()].copy()
    frame = frame[np.abs(frame[target]) > neutral_zone].copy()
    frame["y_recovery"] = (frame[target] > neutral_zone).astype(int)
    frame["destination"] = np.where(frame["y_recovery"] == 1, "positive", "negative")
    frame["origin_regime"] = np.where(frame["bull_probability"] >= 0.5, "bull_side", "bear_side")
    frame["partition"] = np.where(frame["date"] < pd.Timestamp(split_date), "train", "holdout")
    frame["decade"] = (frame["date"].dt.year // 10 * 10).astype(int)
    return frame.sort_values("date").reset_index(drop=True)


def composition(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    groups = [
        ("all", "all", frame),
        *[(partition, "all", part) for partition, part in frame.groupby("partition")],
        *[("all", origin, part) for origin, part in frame.groupby("origin_regime")],
    ]
    for partition, origin, part in groups:
        n = len(part)
        recovery = int(part["y_recovery"].sum())
        rows.append({
            "partition": partition,
            "origin_regime": origin,
            "n": n,
            "n_positive": recovery,
            "n_negative": n - recovery,
            "positive_fraction": recovery / n if n else np.nan,
            "median_future_return_pct": float(part[target].median()) if n else np.nan,
            "mean_future_return_pct": float(part[target].mean()) if n else np.nan,
        })
    for (partition, origin), part in frame.groupby(["partition", "origin_regime"]):
        n = len(part)
        recovery = int(part["y_recovery"].sum())
        rows.append({
            "partition": partition,
            "origin_regime": origin,
            "n": n,
            "n_positive": recovery,
            "n_negative": n - recovery,
            "positive_fraction": recovery / n if n else np.nan,
            "median_future_return_pct": float(part[target].median()),
            "mean_future_return_pct": float(part[target].mean()),
        })
    return pd.DataFrame(rows)


def decade_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decade, part in frame.groupby("decade"):
        n = len(part)
        positive = int(part["y_recovery"].sum())
        rows.append({
            "decade": int(decade),
            "n": n,
            "n_positive": positive,
            "n_negative": n - positive,
            "positive_fraction": positive / n,
            "bear_origin_fraction": float((part["origin_regime"] == "bear_side").mean()),
        })
    return pd.DataFrame(rows)


def baseline_scores(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["partition"] == "train"]
    test = frame[frame["partition"] == "holdout"]
    y_train = train["y_recovery"].to_numpy(int)
    y_test = test["y_recovery"].to_numpy(int)
    train_prior = float(y_train.mean())
    holdout_prior = float(y_test.mean())

    rows = [
        {
            "baseline": "frozen_train_prior",
            "probability_positive": train_prior,
            "holdout_log_loss": log_loss(y_test, train_prior),
            "holdout_brier": brier(y_test, train_prior),
            "holdout_accuracy_at_0_5": float(np.mean((train_prior >= 0.5) == y_test)),
            "holdout_balanced_accuracy": 0.5,
        },
        {
            "baseline": "holdout_prevalence_oracle",
            "probability_positive": holdout_prior,
            "holdout_log_loss": log_loss(y_test, holdout_prior),
            "holdout_brier": brier(y_test, holdout_prior),
            "holdout_accuracy_at_0_5": float(np.mean((holdout_prior >= 0.5) == y_test)),
            "holdout_balanced_accuracy": 0.5,
        },
    ]

    history = list(y_train.astype(float))
    probabilities = []
    for value in y_test:
        probabilities.append(float(np.mean(history)))
        history.append(float(value))
    probabilities = np.asarray(probabilities)
    pred = probabilities >= 0.5
    tpr = float(np.mean(pred[y_test == 1])) if np.any(y_test == 1) else np.nan
    tnr = float(np.mean(~pred[y_test == 0])) if np.any(y_test == 0) else np.nan
    rows.append({
        "baseline": "expanding_historical_prior",
        "probability_positive": np.nan,
        "holdout_log_loss": log_loss(y_test, probabilities),
        "holdout_brier": brier(y_test, probabilities),
        "holdout_accuracy_at_0_5": float(np.mean(pred == y_test)),
        "holdout_balanced_accuracy": float((tpr + tnr) / 2),
    })
    return pd.DataFrame(rows)


def feature_auc_by_origin(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin in ["all", "bear_side", "bull_side"]:
        subset = frame if origin == "all" else frame[frame["origin_regime"] == origin]
        train = subset[subset["partition"] == "train"]
        test = subset[subset["partition"] == "holdout"]
        y_train = train["y_recovery"].to_numpy(int)
        y_test = test["y_recovery"].to_numpy(int)
        for feature in FEATURES:
            raw_train = train[feature].to_numpy(float)
            raw_test = test[feature].to_numpy(float)
            train_auc = roc_auc(y_train, raw_train)
            orientation = 1.0 if np.isnan(train_auc) or train_auc >= 0.5 else -1.0
            rows.append({
                "origin_regime": origin,
                "feature": feature,
                "orientation": "high_positive" if orientation > 0 else "low_positive",
                "train_n": len(train),
                "holdout_n": len(test),
                "holdout_positive_fraction": float(y_test.mean()) if len(y_test) else np.nan,
                "holdout_roc_auc": roc_auc(y_test, orientation * raw_test),
            })
    return pd.DataFrame(rows).sort_values(["origin_regime", "holdout_roc_auc"], ascending=[True, False])


def cluster_summary(frame: pd.DataFrame, max_gap_weeks: int) -> pd.DataFrame:
    dates = frame["date"].sort_values().reset_index(drop=True)
    cluster_ids = []
    cluster = 0
    previous = None
    for date in dates:
        if previous is not None and (date - previous).days > 7 * max_gap_weeks:
            cluster += 1
        cluster_ids.append(cluster)
        previous = date
    tmp = frame.sort_values("date").copy()
    tmp["cluster_id"] = cluster_ids
    rows = []
    for cluster_id, part in tmp.groupby("cluster_id"):
        rows.append({
            "cluster_id": int(cluster_id),
            "start_date": part["date"].min().date(),
            "end_date": part["date"].max().date(),
            "n_episodes": len(part),
            "positive_fraction": float(part["y_recovery"].mean()),
        })
    return pd.DataFrame(rows).sort_values("n_episodes", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path(
            "results/modeling/transition_destination/branch_destination_target_audit__episodes.csv"
        ),
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--neutral-zone", type=float, default=1.0)
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--cluster-gap-weeks", type=int, default=12)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_destination"),
    )
    args = parser.parse_args()

    frame = prepare(args.episodes, args.horizon, args.neutral_zone, args.split_date)
    target = f"future_return_{args.horizon}w"
    comp = composition(frame, target)
    decades = decade_summary(frame)
    baselines = baseline_scores(frame)
    aucs = feature_auc_by_origin(frame)
    clusters = cluster_summary(frame, args.cluster_gap_weeks)

    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.outdir / "branch_destination_structure__task_b_binary_episodes.csv",
        index=False,
    )
    comp.to_csv(
        args.outdir / "branch_destination_structure__origin_partition_summary.csv",
        index=False,
    )
    decades.to_csv(
        args.outdir / "branch_destination_structure__decade_summary.csv",
        index=False,
    )
    baselines.to_csv(
        args.outdir / "branch_destination_structure__prior_baselines.csv",
        index=False,
    )
    aucs.to_csv(
        args.outdir / "branch_destination_structure__feature_auc_by_origin.csv",
        index=False,
    )
    clusters.to_csv(
        args.outdir / "branch_destination_structure__episode_clusters.csv",
        index=False,
    )

    print("Task B origin and partition summary")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPrior-only holdout baselines")
    print(baselines.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nUnivariate holdout AUC by origin regime")
    print(aucs.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nEpisode counts by decade")
    print(decades.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nLargest temporal episode clusters")
    print(clusters.head(15).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
