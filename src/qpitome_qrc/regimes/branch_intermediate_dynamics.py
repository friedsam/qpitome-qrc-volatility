"""Descriptive probes for an intermediate-state interpretation of branch episodes.

The module is intentionally model-free. It reconstructs first-passage escape
from the existing volatility-scaled barriers and summarizes dynamics before and
after entry into the observed branching state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_transition_path import (
    TRANSITION_PATH_COLUMNS,
    add_transition_path_channels,
)
from qpitome_qrc.regimes.branching_state import OutcomeConfig


@dataclass(frozen=True)
class IntermediateProbeConfig:
    max_followup: int = 120
    pre_window: int = 20
    early_window: int = 20
    late_window: int = 10
    min_segment_rows: int = 5


DYNAMIC_METRICS = (
    "return_sign_switch_rate",
    "repair_sign_switch_rate",
    "directional_coherence",
    "rv_ratio_choppiness",
    "mean_downside_pressure",
)


def first_passage(
    running_return: np.ndarray,
    upper: float,
    lower: float,
) -> tuple[int | None, str | None]:
    """Return 1-based first-passage day and event type."""
    upper_hits = np.flatnonzero(running_return >= upper)
    lower_hits = np.flatnonzero(running_return <= lower)
    upper_day = int(upper_hits[0]) + 1 if len(upper_hits) else None
    lower_day = int(lower_hits[0]) + 1 if len(lower_hits) else None
    if upper_day is None and lower_day is None:
        return None, None
    if lower_day is None or (upper_day is not None and upper_day < lower_day):
        return upper_day, "recovery"
    if upper_day is None or lower_day < upper_day:
        return lower_day, "relapse"
    return upper_day, "simultaneous"


def _sign_switch_rate(values: np.ndarray) -> float:
    signs = np.sign(np.asarray(values, dtype=float))
    signs = signs[signs != 0]
    if len(signs) < 2:
        return np.nan
    return float(np.mean(signs[1:] != signs[:-1]))


def segment_metrics(segment: pd.DataFrame) -> dict[str, float]:
    """Summarize directional fluctuation in one causal path segment."""
    missing = set(TRANSITION_PATH_COLUMNS) - set(segment.columns)
    if missing:
        raise KeyError(f"Missing transition channels: {sorted(missing)}")
    signed = segment["signed_return_over_local_vol"].to_numpy(dtype=float)
    repair = segment["drawdown_repair_over_local_vol"].to_numpy(dtype=float)
    rv_change = segment["d_log_rv5_over_rv20"].to_numpy(dtype=float)
    downside = segment["downside_shock_pressure"].to_numpy(dtype=float)
    path_length = float(np.sum(np.abs(signed)))
    coherence = float(abs(np.sum(signed)) / path_length) if path_length > 0 else np.nan
    return {
        "return_sign_switch_rate": _sign_switch_rate(signed),
        "repair_sign_switch_rate": _sign_switch_rate(repair),
        "directional_coherence": coherence,
        "rv_ratio_choppiness": float(np.mean(np.abs(rv_change))),
        "mean_downside_pressure": float(np.mean(downside)),
    }


def build_first_passage_episode_table(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    config: IntermediateProbeConfig | None = None,
) -> pd.DataFrame:
    """Attach fixed-barrier first-passage outcome and residence time to episodes."""
    cfg = config or IntermediateProbeConfig()
    required_daily = {"spy_adj_close"}
    missing = required_daily - set(daily.columns)
    if missing:
        raise KeyError(f"Missing daily columns: {sorted(missing)}")
    required_episode = {"episode_id", "branch_idx", "rv_20d"}
    missing = required_episode - set(episodes.columns)
    if missing:
        raise KeyError(f"Missing episode columns: {sorted(missing)}")

    prices = daily["spy_adj_close"].to_numpy(dtype=float)
    outcome_cfg = OutcomeConfig()
    rows: list[dict[str, object]] = []
    for row in episodes.itertuples(index=False):
        branch_idx = int(row.branch_idx)
        available = min(cfg.max_followup, len(prices) - branch_idx - 1)
        if available < 1:
            continue
        branch_price = prices[branch_idx]
        running_return = prices[branch_idx + 1 : branch_idx + available + 1] / branch_price - 1.0
        scale = float(row.rv_20d) * np.sqrt(OutcomeConfig().horizon / 252.0)
        upper = outcome_cfg.recovery_scale * scale
        lower = -outcome_cfg.relapse_scale * scale
        event_day, event_type = first_passage(running_return, upper, lower)
        rows.append(
            {
                "episode_id": int(row.episode_id),
                "branch_idx": branch_idx,
                "branch_date": getattr(row, "branch_date", pd.NaT),
                "event_day": event_day,
                "event_type": event_type,
                "resolved_within_followup": event_day is not None,
                "upper_barrier": upper,
                "lower_barrier": lower,
            }
        )
    return pd.DataFrame(rows)


def build_intermediate_segment_metrics(
    daily: pd.DataFrame,
    first_passage_episodes: pd.DataFrame,
    config: IntermediateProbeConfig | None = None,
) -> pd.DataFrame:
    """Return pre-entry, early-residence, and late-residence metrics per episode."""
    cfg = config or IntermediateProbeConfig()
    channels = add_transition_path_channels(daily)
    rows: list[dict[str, object]] = []

    for episode in first_passage_episodes.itertuples(index=False):
        if not bool(episode.resolved_within_followup):
            continue
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
        for segment_name, (start, end) in segments.items():
            if start < 0 or end > len(channels) or end - start < cfg.min_segment_rows:
                continue
            segment = channels.iloc[start:end]
            if segment.loc[:, TRANSITION_PATH_COLUMNS].isna().any().any():
                continue
            metrics = segment_metrics(segment)
            rows.append(
                {
                    "episode_id": int(episode.episode_id),
                    "event_type": episode.event_type,
                    "event_day": event_day,
                    "segment": segment_name,
                    "n_rows": len(segment),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Return Cohen-style standardized mean difference with pooled SD."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return 0.0 if np.isclose(np.mean(a), np.mean(b)) else np.inf
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))
