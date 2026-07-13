#!/usr/bin/env python3
"""Cheap gate for whether branch-onset prediction is nontrivial.

This is deliberately not a production classifier. It creates a retrospective,
direction-neutral label from the causal restricted-HMM filtered state sequence:
within ``horizon`` weeks the dominant *long regime* flips (bear={0,1},
bull={2,3}) and the new long regime persists for at least ``min_persist`` of the
next ``persist_window`` weeks.

Each candidate detector uses information available at the forecast week only.
Threshold and orientation are selected on an early chronological training
segment and reported on the untouched later segment. The purpose is to decide
whether onset detection deserves further work or is nearly a trivial threshold
problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
STATE_COLS = [f"hmm4_restricted_filtered_state_{i}" for i in range(4)]


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    yy = y[order].astype(int)
    positives = int(yy.sum())
    if positives == 0:
        return float("nan")
    tp = np.cumsum(yy)
    precision = tp / np.arange(1, len(yy) + 1)
    return float(np.sum(precision * yy) / positives)


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = y.astype(int)
    pred = pred.astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def choose_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        f1 = classification_metrics(y, score >= threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def build_label(
    long_regime: np.ndarray,
    horizon: int,
    persist_window: int,
    min_persist: int,
) -> np.ndarray:
    n = len(long_regime)
    label = np.full(n, np.nan)
    last_usable = n - horizon - persist_window
    for t in range(max(0, last_usable + 1)):
        current = int(long_regime[t])
        transition_found = False
        for lead in range(1, horizon + 1):
            candidate = int(long_regime[t + lead])
            if candidate == current:
                continue
            future = long_regime[t + lead : t + lead + persist_window]
            if int(np.sum(future == candidate)) >= min_persist:
                transition_found = True
                break
        label[t] = float(transition_found)
    return label


def load_frame(path: Path, horizon: int, persist_window: int, min_persist: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "return_pct", *STATE_COLS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    probs = frame[STATE_COLS].to_numpy(float)
    bear_prob = probs[:, 0] + probs[:, 1]
    bull_prob = probs[:, 2] + probs[:, 3]
    long_regime = (bull_prob > bear_prob).astype(int)

    frame["branch_onset"] = build_label(long_regime, horizon, persist_window, min_persist)
    frame["abs_return"] = frame["return_pct"].abs()
    frame["realized_vol_4w"] = frame["return_pct"].rolling(4, min_periods=4).std()
    frame["realized_vol_13w"] = frame["return_pct"].rolling(13, min_periods=8).std()
    frame["vol_change"] = frame["realized_vol_4w"] - frame["realized_vol_13w"]
    frame["long_regime_uncertainty"] = 1.0 - np.abs(bull_prob - bear_prob)
    frame["state_entropy"] = -np.sum(probs * np.log(np.clip(probs, EPS, 1.0)), axis=1)
    frame["max_state_probability"] = probs.max(axis=1)
    frame["one_week_probability_motion"] = np.r_[np.nan, np.abs(np.diff(bull_prob))]
    frame["return_reversal"] = np.r_[np.nan, np.abs(np.diff(np.sign(frame["return_pct"].to_numpy(float))))]
    return frame.dropna().reset_index(drop=True)


def audit_feature(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
) -> dict[str, float | str | int]:
    y_train = train["branch_onset"].to_numpy(int)
    y_test = test["branch_onset"].to_numpy(int)
    raw_train = train[feature].to_numpy(float)
    raw_test = test[feature].to_numpy(float)

    auc_up = roc_auc(y_train, raw_train)
    orientation = 1.0 if np.isnan(auc_up) or auc_up >= 0.5 else -1.0
    train_score = orientation * raw_train
    test_score = orientation * raw_test
    threshold, train_f1 = choose_threshold(y_train, train_score)
    metrics = classification_metrics(y_test, test_score >= threshold)
    return {
        "feature": feature,
        "orientation": "high" if orientation > 0 else "low",
        "train_threshold_oriented": threshold,
        "train_f1": train_f1,
        "test_average_precision": average_precision(y_test, test_score),
        "test_roc_auc": roc_auc(y_test, test_score),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/modeling/weekly_regimes/weekly_regime_baselines/one_step_predictions.csv"),
    )
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--persist-window", type=int, default=4)
    parser.add_argument("--min-persist", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_onset/branch_onset_triviality_audit"),
    )
    args = parser.parse_args()
    if not 0.4 <= args.train_fraction <= 0.8:
        raise ValueError("--train-fraction must be between 0.4 and 0.8")
    if not 1 <= args.min_persist <= args.persist_window:
        raise ValueError("Require 1 <= min_persist <= persist_window")

    frame = load_frame(args.predictions, args.horizon, args.persist_window, args.min_persist)
    split = int(len(frame) * args.train_fraction)
    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()

    features = [
        "abs_return",
        "realized_vol_4w",
        "realized_vol_13w",
        "vol_change",
        "long_regime_uncertainty",
        "state_entropy",
        "max_state_probability",
        "one_week_probability_motion",
        "return_reversal",
    ]
    results = pd.DataFrame([audit_feature(train, test, feature) for feature in features])
    results = results.sort_values("test_average_precision", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame([{
        "n_total": len(frame),
        "train_n": len(train),
        "test_n": len(test),
        "train_start": train["date"].min().date(),
        "train_end": train["date"].max().date(),
        "test_start": test["date"].min().date(),
        "test_end": test["date"].max().date(),
        "train_prevalence": train["branch_onset"].mean(),
        "test_prevalence": test["branch_onset"].mean(),
        "horizon": args.horizon,
        "persist_window": args.persist_window,
        "min_persist": args.min_persist,
    }])

    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / "labeled_weeks.csv", index=False)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    results.to_csv(args.outdir / "trivial_detector_results.csv", index=False)

    print("Branch-onset audit summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nTrivial detector holdout results")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nInterpretation gate")
    print(f"No-skill average precision on test = prevalence = {test['branch_onset'].mean():.5f}")
    print("A detector is interesting only if it materially exceeds this while retaining useful precision and lead-time semantics.")


if __name__ == "__main__":
    main()
