"""Causal branching-state extraction for Phase 3.

This module formalizes the unstable-aftermath morphology discovered in the
exploratory regime analysis. It deliberately separates:

1. causal state detection;
2. episode construction;
3. future outcome labeling.

Future information must never enter steps 1 or 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BranchStateConfig:
    """Operational definition of a candidate unstable-aftermath state.

    Numeric defaults are a mid-grid candidate for audit, not yet the locked
    benchmark definition. The audit runner compares 24 nearby definitions.
    """

    expanding_min_periods: int = 504
    stress_quantile: float = 0.70
    drawdown_threshold: float = -0.06
    rv_ratio_cap: float = 1.00
    stabilization_return_floor: float = -0.01
    prior_decline_threshold: float = -0.04
    prior_decline_lookback: int = 20
    merge_gap_days: int = 3
    min_episode_separation: int = 20


@dataclass(frozen=True)
class OutcomeConfig:
    """Future-resolution labels applied only after episode extraction."""

    horizon: int = 40
    recovery_scale: float = 0.55
    relapse_scale: float = 0.70


STATE_FEATURE_COLUMNS = [
    "rv_20d",
    "branch_stress_cut",
    "drawdown_120d",
    "rv_ratio_5_20_branch",
    "rv_5d_change_5d_branch",
    "return_5d_branch",
    "worst_return_5d_in_prior_window",
]


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing branching-state inputs: {sorted(missing)}")


def add_branch_features(df: pd.DataFrame, config: BranchStateConfig) -> pd.DataFrame:
    """Add causal trailing features used by the branch-state detector."""
    _require_columns(
        df,
        {
            "date",
            "spy_adj_close",
            "spy_log_return",
            "rv_5d",
            "rv_20d",
        },
    )

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["return_5d_branch"] = out["spy_log_return"].rolling(5).sum()
    out["rv_ratio_5_20_branch"] = out["rv_5d"] / out["rv_20d"]
    out["rv_5d_change_5d_branch"] = out["rv_5d"] - out["rv_5d"].shift(5)

    peak_120 = out["spy_adj_close"].rolling(120).max()
    out["drawdown_120d"] = out["spy_adj_close"] / peak_120 - 1.0

    # A decline must actually have occurred recently. We use the worst rolling
    # 5-day return observed in the prior window; this preserves the exploratory
    # intent while making the definition explicit and auditable.
    out["worst_return_5d_in_prior_window"] = (
        out["return_5d_branch"].rolling(config.prior_decline_lookback).min()
    )

    out["branch_stress_cut"] = out["rv_20d"].expanding(
        min_periods=config.expanding_min_periods
    ).quantile(config.stress_quantile)
    return out


def detect_branch_state(
    df: pd.DataFrame,
    config: BranchStateConfig | None = None,
) -> pd.DataFrame:
    """Return rows with a purely causal branching-state mask and component flags."""
    cfg = config or BranchStateConfig()
    out = add_branch_features(df, cfg)

    out["branch_high_stress"] = out["rv_20d"] >= out["branch_stress_cut"]
    out["branch_persistent_damage"] = out["drawdown_120d"] <= cfg.drawdown_threshold
    out["branch_vol_decelerating"] = (
        (out["rv_ratio_5_20_branch"] <= cfg.rv_ratio_cap)
        | (out["rv_5d_change_5d_branch"] < 0.0)
    )
    out["branch_stabilizing"] = (
        out["return_5d_branch"] >= cfg.stabilization_return_floor
    )
    out["branch_prior_decline"] = (
        out["worst_return_5d_in_prior_window"] <= cfg.prior_decline_threshold
    )

    component_cols = [
        "branch_high_stress",
        "branch_persistent_damage",
        "branch_vol_decelerating",
        "branch_stabilizing",
        "branch_prior_decline",
    ]
    out["branch_candidate"] = out[component_cols].all(axis=1)
    return out


def _merge_candidate_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    positives = np.flatnonzero(mask)
    if len(positives) == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(positives[0])
    previous = int(positives[0])
    for current_raw in positives[1:]:
        current = int(current_raw)
        if current - previous - 1 <= max_gap:
            previous = current
            continue
        runs.append((start, previous))
        start = current
        previous = current
    runs.append((start, previous))
    return runs


def build_branch_episodes(
    detected: pd.DataFrame,
    config: BranchStateConfig | None = None,
) -> pd.DataFrame:
    """Merge daily candidates into independent episodes.

    Branch point is the first qualifying day of the merged run. This is the most
    conservative causal anchor and avoids selecting a branch point using future
    information from later within the episode.
    """
    cfg = config or BranchStateConfig()
    if "branch_candidate" not in detected.columns:
        raise KeyError("branch_candidate")

    runs = _merge_candidate_runs(
        detected["branch_candidate"].fillna(False).to_numpy(dtype=bool),
        max_gap=cfg.merge_gap_days,
    )

    accepted: list[tuple[int, int]] = []
    for start, end in runs:
        if not accepted:
            accepted.append((start, end))
            continue
        previous_start, previous_end = accepted[-1]
        if start - previous_start < cfg.min_episode_separation:
            accepted[-1] = (previous_start, max(previous_end, end))
        else:
            accepted.append((start, end))

    rows: list[dict[str, object]] = []
    for episode_id, (start_idx, end_idx) in enumerate(accepted, start=1):
        branch_idx = start_idx
        branch = detected.iloc[branch_idx]
        row: dict[str, object] = {
            "episode_id": episode_id,
            "start_idx": start_idx,
            "branch_idx": branch_idx,
            "end_idx": end_idx,
            "start_date": detected.iloc[start_idx]["date"],
            "branch_date": branch["date"],
            "end_date": detected.iloc[end_idx]["date"],
            "candidate_days": int(
                detected.iloc[start_idx : end_idx + 1]["branch_candidate"].sum()
            ),
            "calendar_rows": int(end_idx - start_idx + 1),
        }
        for col in STATE_FEATURE_COLUMNS:
            row[col] = branch[col]
        rows.append(row)

    return pd.DataFrame(rows)


def label_branch_outcomes(
    detected: pd.DataFrame,
    episodes: pd.DataFrame,
    config: OutcomeConfig | None = None,
) -> pd.DataFrame:
    """Label episode resolution using volatility-scaled future thresholds.

    Recovery threshold:
        + recovery_scale * rv20(branch) * sqrt(h / 252)

    Relapse threshold:
        future running drawdown from branch price <=
        - relapse_scale * rv20(branch) * sqrt(h / 252)

    Recovery takes precedence only when the recovery threshold is reached and the
    relapse threshold is not reached; relapse is symmetric. Episodes that hit
    both or neither are mixed/ambiguous.
    """
    cfg = config or OutcomeConfig()
    if episodes.empty:
        return episodes.copy()

    out = episodes.copy()
    labels: list[str | None] = []
    forward_returns: list[float] = []
    worst_forward_drawdowns: list[float] = []
    thresholds: list[float] = []
    complete: list[bool] = []

    prices = detected["spy_adj_close"].to_numpy(dtype=float)
    n_rows = len(detected)

    for row in out.itertuples(index=False):
        branch_idx = int(row.branch_idx)
        future_end = branch_idx + cfg.horizon
        if future_end >= n_rows:
            labels.append(None)
            forward_returns.append(np.nan)
            worst_forward_drawdowns.append(np.nan)
            thresholds.append(np.nan)
            complete.append(False)
            continue

        branch_price = prices[branch_idx]
        future_prices = prices[branch_idx + 1 : future_end + 1]
        forward_return = float(future_prices[-1] / branch_price - 1.0)
        running_return = future_prices / branch_price - 1.0
        worst_drawdown = float(np.min(running_return))

        rv20 = float(row.rv_20d)
        scale = rv20 * np.sqrt(cfg.horizon / 252.0)
        recovery_threshold = cfg.recovery_scale * scale
        relapse_threshold = -cfg.relapse_scale * scale

        hit_recovery = forward_return >= recovery_threshold
        hit_relapse = worst_drawdown <= relapse_threshold

        if hit_recovery and not hit_relapse:
            label = "recovery"
        elif hit_relapse and not hit_recovery:
            label = "relapse"
        else:
            label = "mixed"

        labels.append(label)
        forward_returns.append(forward_return)
        worst_forward_drawdowns.append(worst_drawdown)
        thresholds.append(scale)
        complete.append(True)

    out["outcome"] = labels
    out["forward_return_h"] = forward_returns
    out["worst_forward_drawdown_h"] = worst_forward_drawdowns
    out["vol_scaled_move_unit"] = thresholds
    out["outcome_complete"] = complete
    out["outcome_horizon"] = cfg.horizon
    return out


def extract_labeled_branch_episodes(
    df: pd.DataFrame,
    state_config: BranchStateConfig | None = None,
    outcome_config: OutcomeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning the daily state frame and labeled episodes."""
    state_cfg = state_config or BranchStateConfig()
    outcome_cfg = outcome_config or OutcomeConfig()
    detected = detect_branch_state(df, state_cfg)
    episodes = build_branch_episodes(detected, state_cfg)
    episodes = label_branch_outcomes(detected, episodes, outcome_cfg)
    return detected, episodes


def config_manifest(
    state_config: BranchStateConfig,
    outcome_config: OutcomeConfig,
) -> dict[str, object]:
    return {
        "state_config": asdict(state_config),
        "outcome_config": asdict(outcome_config),
        "branch_point_rule": "first qualifying day of merged causal candidate run",
        "episode_rule": "merge gaps <= merge_gap_days, then enforce minimum start separation",
    }
