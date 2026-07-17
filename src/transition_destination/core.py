"""Reusable target, episode, path, encoding, and diagnostic logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.binary import binary_metric_row

STATE_COLS = [f"hmm4_restricted_filtered_state_{i}" for i in range(4)]
FEATURES = [
    "return_1w", "momentum_4w", "momentum_13w", "vol_4w", "vol_13w",
    "bull_probability", "long_regime_uncertainty", "one_week_probability_motion",
]
DEFAULT_CHANNELS = [
    "return_pct", "rolling_return_4w", "rolling_return_13w", "rolling_vol_4w",
    "rolling_vol_13w", "bull_probability", "long_regime_uncertainty",
    "one_week_probability_motion",
]
ENCODINGS = {
    "return_only": ["return_pct"],
    "return_uncertainty": ["return_pct", "long_regime_uncertainty"],
    "return_volatility": ["return_pct", "rolling_vol_13w"],
}


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(binary_metric_row("feature", y, score)["roc_auc"])


def load_weekly_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "return_pct", *STATE_COLS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing weekly columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    probabilities = frame[STATE_COLS].to_numpy(float)
    bear = probabilities[:, 0] + probabilities[:, 1]
    bull = probabilities[:, 2] + probabilities[:, 3]
    returns = frame["return_pct"].to_numpy(float)
    frame["return_1w"] = returns
    frame["momentum_4w"] = pd.Series(returns).rolling(4, min_periods=4).sum()
    frame["momentum_13w"] = pd.Series(returns).rolling(13, min_periods=8).sum()
    frame["vol_4w"] = pd.Series(returns).rolling(4, min_periods=4).std()
    frame["vol_13w"] = pd.Series(returns).rolling(13, min_periods=8).std()
    frame["rolling_return_4w"] = frame["momentum_4w"]
    frame["rolling_return_13w"] = frame["momentum_13w"]
    frame["rolling_vol_4w"] = frame["vol_4w"]
    frame["rolling_vol_13w"] = frame["vol_13w"]
    frame["bull_probability"] = bull
    frame["long_regime_uncertainty"] = 1.0 - np.abs(bull - bear)
    frame["one_week_probability_motion"] = np.r_[np.nan, np.abs(np.diff(bull))]
    return frame


def choose_uncertainty_threshold(frame: pd.DataFrame, train_fraction: float) -> tuple[float, int]:
    split = int(len(frame) * train_fraction)
    return float(frame.iloc[:split]["long_regime_uncertainty"].quantile(0.75)), split


def extract_episodes(frame: pd.DataFrame, threshold: float, reset_threshold: float, min_gap: int) -> pd.DataFrame:
    uncertainty = frame["long_regime_uncertainty"].to_numpy(float)
    rows: list[pd.Series] = []
    armed, last_idx = True, -10**9
    for idx in range(1, len(frame)):
        if uncertainty[idx] <= reset_threshold:
            armed = True
        crossed = uncertainty[idx - 1] < threshold <= uncertainty[idx]
        if armed and crossed and idx - last_idx >= min_gap:
            row = frame.iloc[idx].copy()
            row["episode_index"] = idx
            rows.append(row)
            armed, last_idx = False, idx
    return pd.DataFrame(rows).reset_index(drop=True)


def add_future_outcomes(episodes: pd.DataFrame, frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = episodes.copy()
    returns = frame["return_pct"].to_numpy(float)
    for horizon in horizons:
        total, minimum, maximum = [], [], []
        for idx in out["episode_index"].astype(int):
            future = returns[idx + 1 : idx + 1 + horizon]
            if len(future) < horizon:
                total.append(np.nan); minimum.append(np.nan); maximum.append(np.nan)
            else:
                cumulative = np.cumsum(future)
                total.append(float(cumulative[-1]))
                minimum.append(float(cumulative.min()))
                maximum.append(float(cumulative.max()))
        out[f"future_return_{horizon}w"] = total
        out[f"future_min_path_{horizon}w"] = minimum
        out[f"future_max_path_{horizon}w"] = maximum
    return out


def prepare_binary_episodes(path: Path, horizon: int, neutral_zone: float, split_date: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    target = f"future_return_{horizon}w"
    if target not in frame:
        raise ValueError(f"Missing target column {target!r}")
    frame = frame[frame[target].notna() & (np.abs(frame[target]) > neutral_zone)].copy()
    frame["y_positive"] = (frame[target] > neutral_zone).astype(int)
    frame["future_return_pct"] = frame[target].astype(float)
    frame["outcome_available_date"] = frame["date"] + pd.to_timedelta(horizon, unit="W")
    frame["origin_regime"] = np.where(frame["bull_probability"] >= 0.5, "bull_side", "bear_side")
    if split_date is not None:
        frame["partition"] = np.where(frame["date"] < pd.Timestamp(split_date), "train", "holdout")
        frame["decade"] = (frame["date"].dt.year // 10 * 10).astype(int)
    return frame.sort_values("date").reset_index(drop=True)


def target_summary(episodes: pd.DataFrame, horizons: list[int], neutral_zones: list[float], split_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in horizons:
        target = f"future_return_{horizon}w"
        for neutral in neutral_zones:
            usable = episodes[episodes[target].notna()].copy()
            usable["class"] = np.where(usable[target] > neutral, "positive", np.where(usable[target] < -neutral, "negative", "neutral"))
            for partition, part in (("all", usable), ("train", usable[usable["date"] < split_date]), ("holdout", usable[usable["date"] >= split_date])):
                counts = part["class"].value_counts()
                rows.append({
                    "horizon_weeks": horizon, "neutral_zone_pct": neutral, "partition": partition,
                    "n_total": len(part), "n_positive": int(counts.get("positive", 0)),
                    "n_negative": int(counts.get("negative", 0)), "n_neutral": int(counts.get("neutral", 0)),
                    "median_future_return_pct": float(part[target].median()) if len(part) else np.nan,
                })
    return pd.DataFrame(rows)


def feature_auc(episodes: pd.DataFrame, horizon: int, neutral_zone: float, split_date: pd.Timestamp) -> pd.DataFrame:
    target = f"future_return_{horizon}w"
    usable = episodes[episodes[target].notna() & (np.abs(episodes[target]) > neutral_zone)].copy()
    usable["y_positive"] = (usable[target] > neutral_zone).astype(int)
    train, test = usable[usable["date"] < split_date], usable[usable["date"] >= split_date]
    rows = []
    for feature in FEATURES:
        orientation = 1.0 if _auc(train["y_positive"].to_numpy(int), train[feature].to_numpy(float)) >= 0.5 else -1.0
        rows.append({"feature": feature, "orientation": orientation, "train_n": len(train), "holdout_n": len(test), "holdout_roc_auc": _auc(test["y_positive"].to_numpy(int), orientation * test[feature].to_numpy(float))})
    return pd.DataFrame(rows).sort_values("holdout_roc_auc", ascending=False)


def structure_tables(frame: pd.DataFrame, target: str, cluster_gap_weeks: int) -> dict[str, pd.DataFrame]:
    groups = [("all", "all", frame)]
    groups += [(partition, "all", part) for partition, part in frame.groupby("partition")]
    groups += [("all", origin, part) for origin, part in frame.groupby("origin_regime")]
    groups += [(partition, origin, part) for (partition, origin), part in frame.groupby(["partition", "origin_regime"])]
    composition = pd.DataFrame([{
        "partition": partition, "origin_regime": origin, "n": len(part),
        "n_positive": int(part["y_positive"].sum()), "n_negative": int(len(part) - part["y_positive"].sum()),
        "positive_fraction": float(part["y_positive"].mean()),
        "median_future_return_pct": float(part[target].median()),
    } for partition, origin, part in groups if len(part)])
    decades = frame.groupby("decade").agg(n=("y_positive", "size"), n_positive=("y_positive", "sum"), positive_fraction=("y_positive", "mean")).reset_index()
    dates = frame.sort_values("date").copy()
    dates["cluster_id"] = (dates["date"].diff().dt.days.fillna(0) > 7 * cluster_gap_weeks).cumsum()
    clusters = dates.groupby("cluster_id").agg(start_date=("date", "min"), end_date=("date", "max"), n_episodes=("date", "size"), positive_fraction=("y_positive", "mean")).reset_index().sort_values("n_episodes", ascending=False)
    return {"composition": composition, "decades": decades, "clusters": clusters}


def extract_paths(weekly: pd.DataFrame, episodes: pd.DataFrame, channels: list[str], path_weeks: int, split_date: pd.Timestamp) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = weekly[weekly["date"] < split_date][channels].dropna()
    mean = reference.mean().to_numpy(float)
    scale = np.where(reference.std(ddof=0).to_numpy(float) < 1e-8, 1.0, reference.std(ddof=0).to_numpy(float))
    weekly_index = {date: idx for idx, date in enumerate(weekly["date"])}
    paths, metadata_rows, long_rows = [], [], []
    for episode_id, episode in episodes.iterrows():
        end = weekly_index.get(episode["date"])
        if end is None or end - path_weeks + 1 < 0:
            continue
        window = weekly.iloc[end - path_weeks + 1 : end + 1]
        if len(window) != path_weeks or window[channels].isna().any().any():
            continue
        raw = window[channels].to_numpy(float)
        standardized = (raw - mean) / scale
        paths.append(standardized)
        metadata_rows.append({
            "episode_id": int(episode_id), "date": episode["date"], "path_start_date": window["date"].iloc[0],
            "path_end_date": window["date"].iloc[-1], "path_weeks": path_weeks,
            "y_positive": int(episode["y_positive"]), "future_return_pct": float(episode["future_return_pct"]),
            "outcome_available_date": episode["outcome_available_date"], "origin_regime": episode["origin_regime"],
            "momentum_13w_at_episode": float(episode["momentum_13w"]),
        })
        for relative, (_, row) in enumerate(window.iterrows(), start=-(path_weeks - 1)):
            for j, channel in enumerate(channels):
                long_rows.append({"episode_id": int(episode_id), "episode_date": episode["date"], "relative_week": relative, "week_date": row["date"], "channel": channel, "raw_value": float(row[channel]), "standardized_value": float(standardized[relative + path_weeks - 1, j])})
    if not paths:
        raise RuntimeError("No valid episode paths extracted")
    standardization = pd.DataFrame({"channel": channels, "mean": mean, "scale": scale})
    return np.stack(paths), pd.DataFrame(metadata_rows), pd.DataFrame(long_rows), standardization


def encode_rydberg_inputs(paths: np.ndarray, standardization: pd.DataFrame, bound_divisor: float) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    channels = standardization["channel"].tolist()
    channel_to_index = {channel: index for index, channel in enumerate(channels)}
    outputs, rows = {}, []
    for name, selected in ENCODINGS.items():
        missing = sorted(set(selected).difference(channels))
        if missing:
            raise ValueError(f"Required channels absent from path tensor: {missing}")
        tensor = np.tanh(paths[:, :, [channel_to_index[channel] for channel in selected]] / bound_divisor)
        outputs[name] = tensor
        for index, channel in enumerate(selected):
            values = tensor[:, :, index]
            rows.append({"encoding": name, "channel": channel, "minimum": float(values.min()), "median": float(np.median(values)), "maximum": float(values.max()), "fraction_abs_above_0_95": float(np.mean(np.abs(values) > 0.95))})
    return outputs, pd.DataFrame(rows)
