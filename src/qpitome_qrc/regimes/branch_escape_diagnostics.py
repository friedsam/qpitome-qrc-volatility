"""Focused diagnostics for branch-state escape timing, controls, and commitment.

This module contains model-free helpers for three questions:

1. Is residence time associated with eventual escape direction?
2. Are branch episodes dynamically different from matched high-stress non-branch periods?
3. How far before first passage do recovery- and relapse-bound paths separate?

The routines are descriptive and deliberately upstream of hazard, HMM,
Bell-Kramers, or QRC models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    IntermediateProbeConfig,
    build_first_passage_episode_table,
    standardized_mean_difference,
)
from qpitome_qrc.regimes.branch_relaxation_dynamics import (
    RelaxationProbeConfig,
    add_relaxation_channels,
    segment_relaxation_metrics,
)
from qpitome_qrc.regimes.branch_transition_path import add_transition_path_channels
from qpitome_qrc.regimes.branching_state import BranchStateConfig, detect_branch_state


@dataclass(frozen=True)
class EscapeDiagnosticConfig:
    max_followup: int = 120
    control_window: int = 10
    control_exclusion_radius: int = 120
    control_era_years: int = 5
    commitment_max_lead: int = 30
    commitment_required_metrics: int = 2
    commitment_smd_threshold: float = 0.80
    commitment_consecutive_days: int = 3


MATCH_FEATURES = (
    "stress_ratio",
    "drawdown_120d",
    "rv_ratio_5_20_branch",
    "return_5d_branch",
)

COMMITMENT_FEATURES = (
    "signed_return_over_local_vol",
    "downside_shock_pressure",
    "d_log_rv5_over_rv20",
    "drawdown_repair_over_local_vol",
)


def residence_direction_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-passage residence time by escape direction."""
    resolved = events[
        events["resolved_within_followup"]
        & events["event_type"].isin(["recovery", "relapse"])
    ].copy()
    rows: list[dict[str, object]] = []
    for event_type, group in resolved.groupby("event_type"):
        values = group["event_day"].to_numpy(dtype=float)
        rows.append(
            {
                "event_type": event_type,
                "n": len(values),
                "median_event_day": float(np.median(values)),
                "mean_event_day": float(np.mean(values)),
                "q25_event_day": float(np.quantile(values, 0.25)),
                "q75_event_day": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def rank_biserial_recovery_minus_relapse(events: pd.DataFrame) -> float:
    """Pairwise rank-biserial effect for recovery minus relapse residence time."""
    recovery = events.loc[events["event_type"] == "recovery", "event_day"].to_numpy(dtype=float)
    relapse = events.loc[events["event_type"] == "relapse", "event_day"].to_numpy(dtype=float)
    if len(recovery) == 0 or len(relapse) == 0:
        return np.nan
    greater = 0.0
    total = len(recovery) * len(relapse)
    for r in recovery:
        greater += float(np.sum(r > relapse))
        greater += 0.5 * float(np.sum(r == relapse))
    probability = greater / total
    return float(2.0 * probability - 1.0)


def _eligible_control_mask(
    detected: pd.DataFrame,
    episodes: pd.DataFrame,
    cfg: EscapeDiagnosticConfig,
) -> np.ndarray:
    mask = detected["branch_high_stress"].fillna(False).to_numpy(dtype=bool)
    mask &= ~detected["branch_candidate"].fillna(False).to_numpy(dtype=bool)
    n = len(mask)
    for idx in episodes["branch_idx"].astype(int):
        lo = max(0, idx - cfg.control_exclusion_radius)
        hi = min(n, idx + cfg.control_exclusion_radius + 1)
        mask[lo:hi] = False
    mask[:1] = False
    mask[n - cfg.control_window - 1 :] = False
    return mask


def match_high_stress_controls(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    config: EscapeDiagnosticConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nearest-neighbor match branch anchors to high-stress non-branch anchors.

    Matching is descriptive, without replacement, within +/- ``control_era_years``
    when possible. Candidate controls are excluded near any branch episode.
    """
    cfg = config or EscapeDiagnosticConfig()
    detected = detect_branch_state(daily, BranchStateConfig())
    detected = detected.copy()
    detected["stress_ratio"] = detected["rv_20d"] / detected["branch_stress_cut"]
    control_mask = _eligible_control_mask(detected, episodes, cfg)
    candidates = detected.loc[control_mask].copy()
    candidates["anchor_idx"] = candidates.index.astype(int)

    branch = episodes.copy()
    branch = branch.merge(
        detected.loc[:, [*MATCH_FEATURES]].assign(branch_idx=detected.index),
        on="branch_idx",
        how="left",
        suffixes=("", "_detected"),
    )

    pooled = pd.concat(
        [branch.loc[:, MATCH_FEATURES], candidates.loc[:, MATCH_FEATURES]],
        axis=0,
        ignore_index=True,
    )
    center = pooled.mean()
    scale = pooled.std(ddof=0).replace(0.0, 1.0)

    used: set[int] = set()
    rows: list[dict[str, object]] = []
    for row in branch.sort_values("branch_date").itertuples(index=False):
        branch_date = pd.Timestamp(row.branch_date)
        era_lo = branch_date - pd.DateOffset(years=cfg.control_era_years)
        era_hi = branch_date + pd.DateOffset(years=cfg.control_era_years)
        pool = candidates[
            (~candidates["anchor_idx"].isin(used))
            & (candidates["date"] >= era_lo)
            & (candidates["date"] <= era_hi)
        ]
        if pool.empty:
            pool = candidates[~candidates["anchor_idx"].isin(used)]
        if pool.empty:
            break

        branch_vector = (np.asarray([getattr(row, f) for f in MATCH_FEATURES], dtype=float) - center.to_numpy()) / scale.to_numpy()
        pool_matrix = (pool.loc[:, MATCH_FEATURES].to_numpy(dtype=float) - center.to_numpy()) / scale.to_numpy()
        distances = np.sqrt(np.sum((pool_matrix - branch_vector) ** 2, axis=1))
        best_position = int(np.argmin(distances))
        best = pool.iloc[best_position]
        control_idx = int(best["anchor_idx"])
        used.add(control_idx)
        rows.append(
            {
                "episode_id": int(row.episode_id),
                "branch_idx": int(row.branch_idx),
                "branch_date": branch_date,
                "control_idx": control_idx,
                "control_date": best["date"],
                "match_distance": float(distances[best_position]),
            }
        )
    return detected, pd.DataFrame(rows)


def build_matched_control_relaxation(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    matches: pd.DataFrame,
    config: EscapeDiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Compute identical fixed-window relaxation metrics for branch and controls."""
    cfg = config or EscapeDiagnosticConfig()
    relaxation_cfg = RelaxationProbeConfig(min_rows=cfg.control_window)
    channels = add_relaxation_channels(daily)
    event_lookup = events.set_index("episode_id")
    rows: list[dict[str, object]] = []

    for match in matches.itertuples(index=False):
        episode_id = int(match.episode_id)
        if episode_id not in event_lookup.index:
            continue
        event_day = int(event_lookup.loc[episode_id, "event_day"])
        if event_day <= cfg.control_window:
            continue
        for kind, anchor_idx in (
            ("branch", int(match.branch_idx)),
            ("control", int(match.control_idx)),
        ):
            start = anchor_idx + 1
            end = start + cfg.control_window
            segment = channels.iloc[start:end]
            if len(segment) != cfg.control_window:
                continue
            required = ["normalized_return", "log_rv_ratio", "downside_pressure", "drawdown_state"]
            if segment[required].isna().any().any():
                continue
            rows.append(
                {
                    "episode_id": episode_id,
                    "kind": kind,
                    "anchor_idx": anchor_idx,
                    **segment_relaxation_metrics(segment, relaxation_cfg),
                }
            )
    return pd.DataFrame(rows)


def build_resolution_aligned_separation(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    config: EscapeDiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Compute recovery-vs-relapse SMDs at fixed leads before first passage."""
    cfg = config or EscapeDiagnosticConfig()
    channels = add_transition_path_channels(daily)
    rows: list[dict[str, object]] = []
    resolved = events[events["event_type"].isin(["recovery", "relapse"])]

    for lead in range(cfg.commitment_max_lead, 0, -1):
        for feature in COMMITMENT_FEATURES:
            recovery: list[float] = []
            relapse: list[float] = []
            for episode in resolved.itertuples(index=False):
                idx = int(episode.branch_idx) + int(episode.event_day) - lead
                if idx <= int(episode.branch_idx) or idx < 0 or idx >= len(channels):
                    continue
                value = float(channels.iloc[idx][feature])
                if not np.isfinite(value):
                    continue
                if episode.event_type == "recovery":
                    recovery.append(value)
                else:
                    relapse.append(value)
            rows.append(
                {
                    "lead_days_before_escape": lead,
                    "feature": feature,
                    "n_recovery": len(recovery),
                    "n_relapse": len(relapse),
                    "smd_recovery_minus_relapse": standardized_mean_difference(
                        np.asarray(recovery), np.asarray(relapse)
                    ),
                }
            )
    return pd.DataFrame(rows)


def commitment_window(separation: pd.DataFrame, config: EscapeDiagnosticConfig | None = None) -> int | None:
    """Return farthest lead with a sustained multi-feature separation signal."""
    cfg = config or EscapeDiagnosticConfig()
    summary = (
        separation.assign(
            strong=lambda x: np.abs(x["smd_recovery_minus_relapse"])
            >= cfg.commitment_smd_threshold
        )
        .groupby("lead_days_before_escape", as_index=False)["strong"]
        .sum()
        .sort_values("lead_days_before_escape", ascending=False)
    )
    strong_by_lead = {
        int(row.lead_days_before_escape): int(row.strong) >= cfg.commitment_required_metrics
        for row in summary.itertuples(index=False)
    }
    for farthest in range(cfg.commitment_max_lead, cfg.commitment_consecutive_days - 1, -1):
        leads = range(farthest, farthest - cfg.commitment_consecutive_days, -1)
        if all(strong_by_lead.get(lead, False) for lead in leads):
            return farthest
    return None
