#!/usr/bin/env python3
"""Event-time trajectory for Task A branch-onset warning scores.

This is the intended final near-term Task A check. It uses the completed causal
restricted-HMM filtered probabilities, identifies persistent broad-regime flips,
freezes warning thresholds on the first 60% of usable history, and evaluates
whether warning scores rise before holdout transitions.

No HMM refitting and no model training beyond one-dimensional threshold selection
are performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
STATE_COLS = [f"hmm4_restricted_filtered_state_{i}" for i in range(4)]
SCORES = ["long_regime_uncertainty", "one_week_probability_motion"]


def classification_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y = y.astype(int)
    pred = pred.astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def choose_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        f1 = classification_f1(y, score >= threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def build_onset_label(
    broad_regime: np.ndarray,
    horizon: int,
    persist_window: int,
    min_persist: int,
) -> np.ndarray:
    n = len(broad_regime)
    label = np.full(n, np.nan)
    last_usable = n - horizon - persist_window
    for t in range(max(0, last_usable + 1)):
        current = int(broad_regime[t])
        found = False
        for lead in range(1, horizon + 1):
            candidate = int(broad_regime[t + lead])
            if candidate == current:
                continue
            future = broad_regime[t + lead : t + lead + persist_window]
            if int(np.sum(future == candidate)) >= min_persist:
                found = True
                break
        label[t] = float(found)
    return label


def find_persistent_flip_events(
    broad_regime: np.ndarray,
    persist_window: int,
    min_persist: int,
) -> list[int]:
    events: list[int] = []
    n = len(broad_regime)
    for t in range(1, n - persist_window + 1):
        if broad_regime[t] == broad_regime[t - 1]:
            continue
        destination = broad_regime[t]
        future = broad_regime[t : t + persist_window]
        if int(np.sum(future == destination)) >= min_persist:
            events.append(t)
    return events


def load_frame(
    path: Path,
    horizon: int,
    persist_window: int,
    min_persist: int,
) -> pd.DataFrame:
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
    frame["broad_regime"] = (bull_prob > bear_prob).astype(int)
    frame["long_regime_uncertainty"] = 1.0 - np.abs(bull_prob - bear_prob)
    frame["one_week_probability_motion"] = np.r_[np.nan, np.abs(np.diff(bull_prob))]
    frame["branch_onset"] = build_onset_label(
        frame["broad_regime"].to_numpy(int),
        horizon=horizon,
        persist_window=persist_window,
        min_persist=min_persist,
    )
    return frame.dropna().reset_index(drop=True)


def trajectory_rows(
    frame: pd.DataFrame,
    events: list[int],
    thresholds: dict[str, float],
    pre_weeks: int,
    post_weeks: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for event_id, event_idx in enumerate(events):
        event_date = frame.loc[event_idx, "date"]
        origin = int(frame.loc[event_idx - 1, "broad_regime"])
        destination = int(frame.loc[event_idx, "broad_regime"])
        for relative_week in range(-pre_weeks, post_weeks + 1):
            idx = event_idx + relative_week
            if idx < 0 or idx >= len(frame):
                continue
            for score_name in SCORES:
                value = float(frame.loc[idx, score_name])
                rows.append({
                    "event_id": event_id,
                    "event_date": event_date,
                    "origin_regime": origin,
                    "destination_regime": destination,
                    "relative_week": relative_week,
                    "score": score_name,
                    "value": value,
                    "above_threshold": int(value >= thresholds[score_name]),
                })
    return pd.DataFrame(rows)


def summarize_trajectory(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (score, relative_week), part in long.groupby(["score", "relative_week"], sort=True):
        values = part["value"].to_numpy(float)
        rows.append({
            "score": score,
            "relative_week": int(relative_week),
            "n_events": int(part["event_id"].nunique()),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "fraction_above_threshold": float(part["above_threshold"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["score", "relative_week"]).reset_index(drop=True)


def episode_detection_summary(
    long: pd.DataFrame,
    warning_start: int,
    warning_end: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for score, score_frame in long.groupby("score"):
        leads: list[int] = []
        detected = 0
        for _, episode in score_frame.groupby("event_id"):
            window = episode[
                episode["relative_week"].between(warning_start, warning_end)
            ].sort_values("relative_week")
            crossings = window.loc[window["above_threshold"] == 1, "relative_week"]
            if len(crossings):
                detected += 1
                leads.append(int(crossings.iloc[0]))
        n_events = int(score_frame["event_id"].nunique())
        rows.append({
            "score": score,
            "n_events": n_events,
            "detected_events": detected,
            "episode_detection_rate": detected / n_events if n_events else np.nan,
            "median_first_crossing_relative_week": float(np.median(leads)) if leads else np.nan,
            "mean_first_crossing_relative_week": float(np.mean(leads)) if leads else np.nan,
            "warning_window_start": warning_start,
            "warning_window_end": warning_end,
        })
    return pd.DataFrame(rows)


def non_event_false_alarm_rate(
    frame: pd.DataFrame,
    event_indices: list[int],
    thresholds: dict[str, float],
    exclusion_radius: int,
) -> pd.DataFrame:
    excluded = np.zeros(len(frame), dtype=bool)
    for idx in event_indices:
        lo = max(0, idx - exclusion_radius)
        hi = min(len(frame), idx + exclusion_radius + 1)
        excluded[lo:hi] = True
    rows = []
    for score in SCORES:
        values = frame.loc[~excluded, score].to_numpy(float)
        rows.append({
            "score": score,
            "n_non_event_weeks": int(len(values)),
            "non_event_false_alarm_rate": float(np.mean(values >= thresholds[score])),
        })
    return pd.DataFrame(rows)


def maybe_plot(summary: pd.DataFrame, thresholds: dict[str, float], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV trajectory outputs were still written")
        return

    for score in SCORES:
        part = summary[summary["score"] == score].sort_values("relative_week")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = part["relative_week"].to_numpy(int)
        median = part["median"].to_numpy(float)
        q25 = part["q25"].to_numpy(float)
        q75 = part["q75"].to_numpy(float)
        ax.plot(x, median, marker="o", label="median across events")
        ax.fill_between(x, q25, q75, alpha=0.2, label="interquartile range")
        ax.axhline(thresholds[score], linestyle="--", label="frozen train threshold")
        ax.axvline(0, linestyle=":", label="persistent regime flip")
        ax.set_xlabel("Weeks relative to persistent broad-regime flip")
        ax.set_ylabel(score.replace("_", " "))
        ax.set_title(f"Task A event-time trajectory: {score.replace('_', ' ')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path / f"trajectory_{score}.png", dpi=180)
        plt.close(fig)


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
    parser.add_argument("--pre-weeks", type=int, default=8)
    parser.add_argument("--post-weeks", type=int, default=2)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_onset/branch_onset_trajectory"),
    )
    args = parser.parse_args()

    frame = load_frame(
        args.predictions,
        horizon=args.horizon,
        persist_window=args.persist_window,
        min_persist=args.min_persist,
    )
    split = int(len(frame) * args.train_fraction)
    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy().reset_index(drop=True)

    y_train = train["branch_onset"].to_numpy(int)
    thresholds = {
        score: choose_threshold(y_train, train[score].to_numpy(float))
        for score in SCORES
    }

    test_events = find_persistent_flip_events(
        test["broad_regime"].to_numpy(int),
        persist_window=args.persist_window,
        min_persist=args.min_persist,
    )
    test_events = [idx for idx in test_events if idx >= args.pre_weeks]
    if not test_events:
        raise RuntimeError("No holdout events remain after event-time window requirements")

    long = trajectory_rows(
        test,
        test_events,
        thresholds=thresholds,
        pre_weeks=args.pre_weeks,
        post_weeks=args.post_weeks,
    )
    trajectory = summarize_trajectory(long)
    episode = episode_detection_summary(long, warning_start=-args.horizon, warning_end=-1)
    false_alarms = non_event_false_alarm_rate(
        test,
        test_events,
        thresholds=thresholds,
        exclusion_radius=args.pre_weeks,
    )
    threshold_frame = pd.DataFrame([
        {"score": score, "frozen_train_threshold": threshold}
        for score, threshold in thresholds.items()
    ])
    event_frame = pd.DataFrame([
        {
            "event_id": event_id,
            "event_index_in_holdout": idx,
            "event_date": test.loc[idx, "date"],
            "origin_regime": int(test.loc[idx - 1, "broad_regime"]),
            "destination_regime": int(test.loc[idx, "broad_regime"]),
        }
        for event_id, idx in enumerate(test_events)
    ])

    args.outdir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.outdir / "event_time_scores_long.csv", index=False)
    trajectory.to_csv(args.outdir / "event_time_trajectory_summary.csv", index=False)
    episode.to_csv(args.outdir / "episode_detection_summary.csv", index=False)
    false_alarms.to_csv(args.outdir / "non_event_false_alarm_summary.csv", index=False)
    threshold_frame.to_csv(args.outdir / "frozen_thresholds.csv", index=False)
    event_frame.to_csv(args.outdir / "holdout_events.csv", index=False)
    maybe_plot(trajectory, thresholds, args.outdir)

    print("Frozen training thresholds")
    print(threshold_frame.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nHoldout event count")
    print(len(test_events))
    print("\nEpisode-level warning performance during weeks -4 through -1")
    print(episode.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNon-event false-alarm rates")
    print(false_alarms.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nEvent-time trajectory")
    print(trajectory.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
