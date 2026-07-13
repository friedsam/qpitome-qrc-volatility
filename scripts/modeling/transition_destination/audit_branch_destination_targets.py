#!/usr/bin/env python3
"""Cheap Task B target audit for branch destination prediction.

Episodes are admitted causally when restricted-HMM broad-regime uncertainty
crosses a threshold frozen on the early training period. One observation is kept
per uncertainty episode. Outcomes are future observed cumulative weekly returns,
not future HMM labels.

The script reports sample counts, class balance, neutral-zone sensitivity, and
holdout univariate predictability for simple causal features. It is a target
selection gate, not the final classical baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATE_COLS = [f"hmm4_restricted_filtered_state_{i}" for i in range(4)]
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


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    yy = y[order].astype(int)
    positives = int(yy.sum())
    if positives == 0:
        return float("nan")
    tp = np.cumsum(yy)
    precision = tp / np.arange(1, len(yy) + 1)
    return float(np.sum(precision * yy) / positives)


def choose_uncertainty_threshold(frame: pd.DataFrame, train_fraction: float) -> tuple[float, int]:
    split = int(len(frame) * train_fraction)
    train = frame.iloc[:split]
    # Freeze a simple high-uncertainty threshold from early history. Using the
    # 75th percentile avoids importing future destination information.
    threshold = float(train["long_regime_uncertainty"].quantile(0.75))
    return threshold, split


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "return_pct", *STATE_COLS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    probs = frame[STATE_COLS].to_numpy(float)
    bear = probs[:, 0] + probs[:, 1]
    bull = probs[:, 2] + probs[:, 3]
    ret = frame["return_pct"].to_numpy(float)

    frame["return_1w"] = ret
    frame["momentum_4w"] = pd.Series(ret).rolling(4, min_periods=4).sum().to_numpy()
    frame["momentum_13w"] = pd.Series(ret).rolling(13, min_periods=8).sum().to_numpy()
    frame["vol_4w"] = pd.Series(ret).rolling(4, min_periods=4).std().to_numpy()
    frame["vol_13w"] = pd.Series(ret).rolling(13, min_periods=8).std().to_numpy()
    frame["bull_probability"] = bull
    frame["long_regime_uncertainty"] = 1.0 - np.abs(bull - bear)
    frame["one_week_probability_motion"] = np.r_[np.nan, np.abs(np.diff(bull))]
    return frame.dropna().reset_index(drop=True)


def extract_episodes(
    frame: pd.DataFrame,
    threshold: float,
    reset_threshold: float,
    min_gap: int,
) -> pd.DataFrame:
    uncertainty = frame["long_regime_uncertainty"].to_numpy(float)
    rows: list[pd.Series] = []
    armed = True
    last_idx = -10**9

    for idx in range(1, len(frame)):
        if uncertainty[idx] <= reset_threshold:
            armed = True
        crossed = uncertainty[idx - 1] < threshold <= uncertainty[idx]
        if armed and crossed and idx - last_idx >= min_gap:
            row = frame.iloc[idx].copy()
            row["episode_index"] = idx
            rows.append(row)
            last_idx = idx
            armed = False

    if not rows:
        return pd.DataFrame(columns=list(frame.columns) + ["episode_index"])
    return pd.DataFrame(rows).reset_index(drop=True)


def add_future_outcomes(episodes: pd.DataFrame, frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = episodes.copy()
    returns = frame["return_pct"].to_numpy(float)
    for horizon in horizons:
        values = []
        minimum_path = []
        maximum_path = []
        for idx in out["episode_index"].astype(int):
            future = returns[idx + 1 : idx + 1 + horizon]
            if len(future) < horizon:
                values.append(np.nan)
                minimum_path.append(np.nan)
                maximum_path.append(np.nan)
                continue
            cumulative = np.cumsum(future)
            values.append(float(cumulative[-1]))
            minimum_path.append(float(cumulative.min()))
            maximum_path.append(float(cumulative.max()))
        out[f"future_return_{horizon}w"] = values
        out[f"future_min_path_{horizon}w"] = minimum_path
        out[f"future_max_path_{horizon}w"] = maximum_path
    return out


def target_summary(
    episodes: pd.DataFrame,
    horizons: list[int],
    neutral_zones: list[float],
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in horizons:
        target_col = f"future_return_{horizon}w"
        for neutral in neutral_zones:
            usable = episodes[episodes[target_col].notna()].copy()
            usable["class"] = np.where(
                usable[target_col] > neutral,
                "recovery",
                np.where(usable[target_col] < -neutral, "deterioration", "neutral"),
            )
            for partition, part in [
                ("all", usable),
                ("train", usable[usable["date"] < split_date]),
                ("holdout", usable[usable["date"] >= split_date]),
            ]:
                counts = part["class"].value_counts()
                binary_n = int(counts.get("recovery", 0) + counts.get("deterioration", 0))
                rows.append({
                    "horizon_weeks": horizon,
                    "neutral_zone_pct": neutral,
                    "partition": partition,
                    "n_total": int(len(part)),
                    "n_recovery": int(counts.get("recovery", 0)),
                    "n_deterioration": int(counts.get("deterioration", 0)),
                    "n_neutral": int(counts.get("neutral", 0)),
                    "binary_n": binary_n,
                    "recovery_fraction_binary": (
                        float(counts.get("recovery", 0) / binary_n) if binary_n else np.nan
                    ),
                    "median_future_return_pct": float(part[target_col].median()) if len(part) else np.nan,
                })
    return pd.DataFrame(rows)


def trivial_predictability(
    episodes: pd.DataFrame,
    horizon: int,
    neutral_zone: float,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    target_col = f"future_return_{horizon}w"
    usable = episodes[episodes[target_col].notna()].copy()
    usable = usable[np.abs(usable[target_col]) > neutral_zone].copy()
    usable["y"] = (usable[target_col] > neutral_zone).astype(int)
    train = usable[usable["date"] < split_date]
    test = usable[usable["date"] >= split_date]

    rows: list[dict] = []
    for feature in FEATURES:
        train_score = train[feature].to_numpy(float)
        test_score = test[feature].to_numpy(float)
        y_train = train["y"].to_numpy(int)
        y_test = test["y"].to_numpy(int)
        auc_train = roc_auc(y_train, train_score)
        orientation = 1.0 if np.isnan(auc_train) or auc_train >= 0.5 else -1.0
        oriented_test = orientation * test_score
        rows.append({
            "feature": feature,
            "orientation": "high_recovery" if orientation > 0 else "low_recovery",
            "train_n": int(len(train)),
            "holdout_n": int(len(test)),
            "holdout_prevalence_recovery": float(y_test.mean()) if len(y_test) else np.nan,
            "holdout_roc_auc": roc_auc(y_test, oriented_test),
            "holdout_average_precision": average_precision(y_test, oriented_test),
        })
    return pd.DataFrame(rows).sort_values("holdout_roc_auc", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/modeling/weekly_regimes/weekly_regime_baselines/one_step_predictions.csv"
        ),
    )
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--reset-threshold", type=float, default=0.35)
    parser.add_argument("--min-gap", type=int, default=6)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--neutral-zones", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--audit-horizon", type=int, default=8)
    parser.add_argument("--audit-neutral-zone", type=float, default=1.0)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_destination"),
    )
    args = parser.parse_args()

    frame = load_predictions(args.predictions)
    threshold, split = choose_uncertainty_threshold(frame, args.train_fraction)
    split_date = frame.iloc[split]["date"]
    episodes = extract_episodes(
        frame,
        threshold=threshold,
        reset_threshold=args.reset_threshold,
        min_gap=args.min_gap,
    )
    episodes = add_future_outcomes(episodes, frame, args.horizons)

    summaries = target_summary(
        episodes,
        horizons=args.horizons,
        neutral_zones=args.neutral_zones,
        split_date=split_date,
    )
    trivial = trivial_predictability(
        episodes,
        horizon=args.audit_horizon,
        neutral_zone=args.audit_neutral_zone,
        split_date=split_date,
    )
    metadata = pd.DataFrame([{
        "n_weekly_rows": len(frame),
        "n_episodes": len(episodes),
        "frozen_uncertainty_threshold": threshold,
        "reset_threshold": args.reset_threshold,
        "min_gap_weeks": args.min_gap,
        "train_fraction": args.train_fraction,
        "split_date": split_date.date(),
        "audit_horizon_weeks": args.audit_horizon,
        "audit_neutral_zone_pct": args.audit_neutral_zone,
    }])

    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(
        args.outdir / "branch_destination_target_audit__metadata.csv",
        index=False,
    )
    episodes.to_csv(
        args.outdir / "branch_destination_target_audit__episodes.csv",
        index=False,
    )
    summaries.to_csv(
        args.outdir / "branch_destination_target_audit__target_summary.csv",
        index=False,
    )
    trivial.to_csv(
        args.outdir / "branch_destination_target_audit__trivial_predictability.csv",
        index=False,
    )

    print("Task B target audit metadata")
    print(metadata.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nTarget counts and balance")
    print(summaries.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(
        f"\nTrivial holdout predictability for {args.audit_horizon}-week destination "
        f"with +/-{args.audit_neutral_zone:.1f}% neutral zone"
    )
    print(trivial.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
