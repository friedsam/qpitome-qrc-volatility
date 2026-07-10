"""Model-free relaxation diagnostics around observed branch-state entry.

This module tests a narrow hypothesis: internal relaxation may slow enough during
branch episodes that a normally transient intermediate becomes observable at the
fixed daily market sampling scale.

The functions here are descriptive only. They do not fit hazard, HMM, Bell-
Kramers, or reservoir models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    IntermediateProbeConfig,
    build_first_passage_episode_table,
)


@dataclass(frozen=True)
class RelaxationProbeConfig:
    pre_window: int = 20
    early_window: int = 20
    late_window: int = 10
    max_followup: int = 120
    min_rows: int = 8
    shock_threshold_sigma: float = -1.0
    shock_baseline_window: int = 5
    shock_lags: tuple[int, ...] = (1, 3, 5)


RELAXATION_VARIABLES = (
    "log_rv_ratio",
    "downside_pressure",
    "drawdown_state",
)


def add_relaxation_channels(daily: pd.DataFrame) -> pd.DataFrame:
    """Construct fixed causal state channels used only for relaxation probes."""
    required = {"spy_adj_close", "spy_log_return", "rv_5d", "rv_20d"}
    missing = required - set(daily.columns)
    if missing:
        raise KeyError(f"Missing relaxation inputs: {sorted(missing)}")

    out = daily.copy().sort_values("date").reset_index(drop=True)
    local_daily_vol = out["rv_20d"] / np.sqrt(252.0)
    safe_daily_vol = local_daily_vol.where(local_daily_vol > 0)
    out["normalized_return"] = out["spy_log_return"] / safe_daily_vol
    out["log_rv_ratio"] = np.log(out["rv_5d"] / out["rv_20d"])

    negative_energy = out["spy_log_return"].clip(upper=0.0).pow(2)
    total_energy = out["spy_log_return"].pow(2)
    out["downside_pressure"] = (
        negative_energy.rolling(5, min_periods=5).sum()
        / total_energy.rolling(20, min_periods=20).sum().where(lambda x: x > 0)
    )

    peak_120 = out["spy_adj_close"].rolling(120, min_periods=120).max()
    out["drawdown_state"] = out["spy_adj_close"] / peak_120 - 1.0
    return out


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Lag-1 correlation after removing a linear trend within the segment."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    residual = x - (intercept + slope * t)
    if np.std(residual[:-1]) == 0 or np.std(residual[1:]) == 0:
        return np.nan
    return float(np.corrcoef(residual[:-1], residual[1:])[0, 1])


def restoring_fraction(values: np.ndarray) -> float:
    """Fraction of steps that move a deviation back toward the segment median."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    center = float(np.median(x))
    deviation = x[:-1] - center
    delta = np.diff(x)
    informative = deviation != 0
    if not np.any(informative):
        return np.nan
    restoring = deviation[informative] * delta[informative] < 0
    return float(np.mean(restoring))


def same_side_run_length(values: np.ndarray) -> float:
    """Mean run length on one side of the segment median."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    center = float(np.median(x))
    signs = np.sign(x - center)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return np.nan
    lengths: list[int] = []
    current = 1
    for previous, value in zip(signs[:-1], signs[1:]):
        if value == previous:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return float(np.mean(lengths))


def shock_carryover(
    normalized_return: np.ndarray,
    response: np.ndarray,
    threshold_sigma: float = -1.0,
    baseline_window: int = 5,
    lags: tuple[int, ...] = (1, 3, 5),
) -> dict[int, float]:
    """Median normalized response remaining after local downside shocks.

    For each shock day, response excess is measured relative to the mean of the
    preceding baseline window. The lag-k excess is divided by lag-1 excess when
    the lag-1 response is positive. Values near one indicate persistent response;
    values near zero indicate decay toward baseline.
    """
    ret = np.asarray(normalized_return, dtype=float)
    y = np.asarray(response, dtype=float)
    if len(ret) != len(y):
        raise ValueError("normalized_return and response must have equal length")

    ratios: dict[int, list[float]] = {lag: [] for lag in lags}
    max_lag = max(lags)
    for i in range(baseline_window, len(ret) - max_lag):
        if not np.isfinite(ret[i]) or ret[i] > threshold_sigma:
            continue
        baseline = y[i - baseline_window : i]
        if not np.isfinite(baseline).all():
            continue
        base = float(np.mean(baseline))
        lag1_excess = y[i + 1] - base
        if not np.isfinite(lag1_excess) or lag1_excess <= 0:
            continue
        for lag in lags:
            excess = y[i + lag] - base
            if np.isfinite(excess):
                ratios[lag].append(float(excess / lag1_excess))

    return {
        lag: float(np.median(values)) if values else np.nan
        for lag, values in ratios.items()
    }


def segment_relaxation_metrics(segment: pd.DataFrame, config: RelaxationProbeConfig) -> dict[str, float]:
    """Compute persistence, restoring, run-length, and shock-decay metrics."""
    required = {"normalized_return", *RELAXATION_VARIABLES}
    missing = required - set(segment.columns)
    if missing:
        raise KeyError(f"Missing relaxation channels: {sorted(missing)}")

    metrics: dict[str, float] = {}
    for variable in RELAXATION_VARIABLES:
        values = segment[variable].to_numpy(dtype=float)
        metrics[f"{variable}__lag1"] = lag1_autocorrelation(values)
        metrics[f"{variable}__restoring_fraction"] = restoring_fraction(values)
        metrics[f"{variable}__same_side_run_length"] = same_side_run_length(values)

    carryover = shock_carryover(
        segment["normalized_return"].to_numpy(dtype=float),
        segment["log_rv_ratio"].to_numpy(dtype=float),
        threshold_sigma=config.shock_threshold_sigma,
        baseline_window=config.shock_baseline_window,
        lags=config.shock_lags,
    )
    for lag, value in carryover.items():
        metrics[f"log_rv_ratio__shock_carryover_lag{lag}"] = value
    return metrics


def build_relaxation_segment_table(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    config: RelaxationProbeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return first-passage event table and per-segment relaxation metrics."""
    cfg = config or RelaxationProbeConfig()
    intermediate_cfg = IntermediateProbeConfig(max_followup=cfg.max_followup)
    events = build_first_passage_episode_table(daily, episodes, intermediate_cfg)
    events = events[events["resolved_within_followup"]].copy()
    channels = add_relaxation_channels(daily)

    rows: list[dict[str, object]] = []
    for episode in events.itertuples(index=False):
        branch_idx = int(episode.branch_idx)
        event_day = int(episode.event_day)
        escape_idx = branch_idx + event_day
        segments = {
            "pre_entry": (branch_idx - cfg.pre_window, branch_idx),
            "early_residence": (
                branch_idx + 1,
                min(escape_idx, branch_idx + 1 + cfg.early_window),
            ),
            "late_residence": (
                max(branch_idx + 1, escape_idx - cfg.late_window),
                escape_idx,
            ),
        }
        for name, (start, end) in segments.items():
            if start < 0 or end > len(channels) or end - start < cfg.min_rows:
                continue
            segment = channels.iloc[start:end]
            if segment[["normalized_return", *RELAXATION_VARIABLES]].isna().any().any():
                continue
            rows.append(
                {
                    "episode_id": int(episode.episode_id),
                    "event_type": episode.event_type,
                    "event_day": event_day,
                    "segment": name,
                    "n_rows": len(segment),
                    **segment_relaxation_metrics(segment, cfg),
                }
            )
    return events, pd.DataFrame(rows)
