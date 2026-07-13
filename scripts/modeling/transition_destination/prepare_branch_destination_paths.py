#!/usr/bin/env python3
"""Prepare causal episode paths for Task B reservoir experiments.

The script joins the frozen Task B episode table to the weekly causal HMM output
and extracts a fixed-length sequence ending at each episode date. The sequence
contains only information available by that date.

Outputs are deliberately model-agnostic so the same arrays can be consumed by a
matched classical reservoir and a later Rydberg reservoir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATE_COLS = [f"hmm4_restricted_filtered_state_{i}" for i in range(4)]
DEFAULT_CHANNELS = [
    "return_pct",
    "rolling_return_4w",
    "rolling_return_13w",
    "rolling_vol_4w",
    "rolling_vol_13w",
    "bull_probability",
    "long_regime_uncertainty",
    "one_week_probability_motion",
]


def load_weekly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "return_pct", *STATE_COLS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing weekly columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    probabilities = frame[STATE_COLS].to_numpy(float)
    bear_probability = probabilities[:, 0] + probabilities[:, 1]
    bull_probability = probabilities[:, 2] + probabilities[:, 3]
    returns = frame["return_pct"].to_numpy(float)

    frame["rolling_return_4w"] = pd.Series(returns).rolling(4, min_periods=4).sum()
    frame["rolling_return_13w"] = pd.Series(returns).rolling(13, min_periods=8).sum()
    frame["rolling_vol_4w"] = pd.Series(returns).rolling(4, min_periods=4).std()
    frame["rolling_vol_13w"] = pd.Series(returns).rolling(13, min_periods=8).std()
    frame["bull_probability"] = bull_probability
    frame["long_regime_uncertainty"] = 1.0 - np.abs(bull_probability - bear_probability)
    frame["one_week_probability_motion"] = np.r_[np.nan, np.abs(np.diff(bull_probability))]
    return frame


def load_episodes(path: Path, horizon: int, neutral_zone: float) -> pd.DataFrame:
    episodes = pd.read_csv(path)
    episodes["date"] = pd.to_datetime(episodes["date"])
    target = f"future_return_{horizon}w"
    if target not in episodes:
        raise ValueError(f"Missing episode target column {target!r}")
    episodes = episodes[episodes[target].notna()].copy()
    episodes = episodes[np.abs(episodes[target]) > neutral_zone].copy()
    episodes["y_positive"] = (episodes[target] > neutral_zone).astype(int)
    episodes["future_return_pct"] = episodes[target].astype(float)
    episodes["outcome_available_date"] = episodes["date"] + pd.to_timedelta(horizon, unit="W")
    return episodes.sort_values("date").reset_index(drop=True)


def standardization_reference(
    weekly: pd.DataFrame,
    channels: list[str],
    split_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    reference = weekly[weekly["date"] < split_date][channels].dropna()
    if reference.empty:
        raise ValueError("No pre-split weekly rows available for standardization")
    mean = reference.mean(axis=0).to_numpy(float)
    scale = reference.std(axis=0, ddof=0).to_numpy(float)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def extract_paths(
    weekly: pd.DataFrame,
    episodes: pd.DataFrame,
    channels: list[str],
    path_weeks: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    weekly_index = {date: idx for idx, date in enumerate(weekly["date"])}
    paths: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    long_rows: list[dict] = []

    for episode_id, episode in episodes.iterrows():
        date = episode["date"]
        if date not in weekly_index:
            continue
        end_idx = weekly_index[date]
        start_idx = end_idx - path_weeks + 1
        if start_idx < 0:
            continue
        window = weekly.iloc[start_idx : end_idx + 1].copy()
        if len(window) != path_weeks or window[channels].isna().any().any():
            continue
        raw = window[channels].to_numpy(float)
        standardized = (raw - mean) / scale
        paths.append(standardized)

        metadata_rows.append({
            "episode_id": int(episode_id),
            "date": date,
            "path_start_date": window["date"].iloc[0],
            "path_end_date": window["date"].iloc[-1],
            "path_weeks": path_weeks,
            "y_positive": int(episode["y_positive"]),
            "future_return_pct": float(episode["future_return_pct"]),
            "outcome_available_date": episode["outcome_available_date"],
            "origin_regime": "bull_side" if float(episode["bull_probability"]) >= 0.5 else "bear_side",
            "momentum_13w_at_episode": float(episode["momentum_13w"]),
            "bull_probability_at_episode": float(episode["bull_probability"]),
            "long_regime_uncertainty_at_episode": float(episode["long_regime_uncertainty"]),
        })

        for relative_week, (_, row) in enumerate(window.iterrows(), start=-(path_weeks - 1)):
            for channel_idx, channel in enumerate(channels):
                long_rows.append({
                    "episode_id": int(episode_id),
                    "episode_date": date,
                    "relative_week": relative_week,
                    "week_date": row["date"],
                    "channel": channel,
                    "raw_value": float(row[channel]),
                    "standardized_value": float(standardized[relative_week + path_weeks - 1, channel_idx]),
                })

    if not paths:
        raise RuntimeError("No valid episode paths extracted")
    return np.stack(paths), pd.DataFrame(metadata_rows), pd.DataFrame(long_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weekly-predictions",
        type=Path,
        default=Path("results/modeling/weekly_regimes/weekly_regime_baselines/one_step_predictions.csv"),
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("results/modeling/transition_destination/branch_destination_target_audit/episodes.csv"),
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--neutral-zone", type=float, default=1.0)
    parser.add_argument("--path-weeks", type=int, default=13)
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_destination/branch_destination_paths"),
    )
    args = parser.parse_args()

    weekly = load_weekly(args.weekly_predictions)
    missing_channels = sorted(set(args.channels).difference(weekly.columns))
    if missing_channels:
        raise ValueError(f"Unknown channels: {missing_channels}")
    episodes = load_episodes(args.episodes, args.horizon, args.neutral_zone)
    split_date = pd.Timestamp(args.split_date)
    mean, scale = standardization_reference(weekly, args.channels, split_date)
    paths, metadata, long = extract_paths(
        weekly,
        episodes,
        channels=args.channels,
        path_weeks=args.path_weeks,
        mean=mean,
        scale=scale,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / "paths.npy", paths)
    metadata.to_csv(args.outdir / "episode_metadata.csv", index=False)
    long.to_csv(args.outdir / "paths_long.csv", index=False)
    pd.DataFrame({"channel": args.channels, "mean": mean, "scale": scale}).to_csv(
        args.outdir / "standardization.csv", index=False
    )
    manifest = {
        "n_episodes": int(len(metadata)),
        "n_positive": int(metadata["y_positive"].sum()),
        "n_negative": int(len(metadata) - metadata["y_positive"].sum()),
        "path_weeks": int(args.path_weeks),
        "n_channels": int(len(args.channels)),
        "channels": args.channels,
        "array_shape": list(paths.shape),
        "horizon_weeks": int(args.horizon),
        "neutral_zone_pct": float(args.neutral_zone),
        "split_date": str(split_date.date()),
        "standardization": "fixed mean and scale from weekly observations before split date",
        "causality": "each path ends at episode date and includes no future observations",
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Task B reservoir path dataset")
    print(json.dumps(manifest, indent=2))
    print("\nEpisode partitions")
    summary = metadata.assign(
        partition=np.where(metadata["date"] < split_date, "train", "holdout")
    ).groupby(["partition", "origin_regime"])["y_positive"].agg(["count", "sum", "mean"])
    print(summary.to_string(float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
