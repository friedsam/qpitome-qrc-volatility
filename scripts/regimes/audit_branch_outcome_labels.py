"""Audit recovery/relapse/mixed outcome labels before locking them.

This runner does not change the canonical detector or labels. It asks:

1. Are mixed episodes mixed because both thresholds are reached or neither?
2. How many labels would change if recovery used maximum forward return rather
   than terminal horizon return?
3. Which episodes sit close to the current thresholds?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branching_state import (
    BranchStateConfig,
    OutcomeConfig,
    extract_labeled_branch_episodes,
)


DEFAULT_EXTENDED_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_FROZEN_DATA = Path("data/processed/phase2_spy_vix_volatility.csv")
DEFAULT_OUTPUT = Path("results/regimes/branch_outcome_label_audit_v1")

STATE_CONFIG = BranchStateConfig(
    stress_quantile=0.70,
    drawdown_threshold=-0.06,
    rv_ratio_cap=1.00,
    stabilization_return_floor=-0.005,
    prior_decline_threshold=-0.05,
    prior_decline_lookback=20,
    merge_gap_days=3,
    min_episode_separation=20,
)
OUTCOME_CONFIG = OutcomeConfig(horizon=40, recovery_scale=0.55, relapse_scale=0.70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_data_path(requested: Path | None) -> Path:
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(requested)
        return requested
    if DEFAULT_EXTENDED_DATA.exists():
        return DEFAULT_EXTENDED_DATA
    if DEFAULT_FROZEN_DATA.exists():
        return DEFAULT_FROZEN_DATA
    raise FileNotFoundError


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["date"])
    daily, episodes = extract_labeled_branch_episodes(df, STATE_CONFIG, OUTCOME_CONFIG)
    prices = daily["spy_adj_close"].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    for ep in episodes[episodes["outcome_complete"]].itertuples(index=False):
        branch_idx = int(ep.branch_idx)
        future_end = branch_idx + OUTCOME_CONFIG.horizon
        branch_price = prices[branch_idx]
        future_returns = prices[branch_idx + 1 : future_end + 1] / branch_price - 1.0

        move_unit = float(ep.vol_scaled_move_unit)
        recovery_threshold = OUTCOME_CONFIG.recovery_scale * move_unit
        relapse_threshold = -OUTCOME_CONFIG.relapse_scale * move_unit
        terminal_return = float(future_returns[-1])
        max_forward_return = float(np.max(future_returns))
        worst_forward_drawdown = float(np.min(future_returns))

        hit_terminal_recovery = terminal_return >= recovery_threshold
        hit_anytime_recovery = max_forward_return >= recovery_threshold
        hit_relapse = worst_forward_drawdown <= relapse_threshold

        if hit_terminal_recovery and hit_relapse:
            mixed_subtype = "both"
        elif not hit_terminal_recovery and not hit_relapse:
            mixed_subtype = "neither"
        else:
            mixed_subtype = "not_mixed"

        if hit_anytime_recovery and not hit_relapse:
            anytime_label = "recovery"
        elif hit_relapse and not hit_anytime_recovery:
            anytime_label = "relapse"
        else:
            anytime_label = "mixed"

        rows.append(
            {
                "episode_id": ep.episode_id,
                "branch_date": ep.branch_date,
                "current_outcome": ep.outcome,
                "mixed_subtype": mixed_subtype,
                "recovery_threshold": recovery_threshold,
                "relapse_threshold": relapse_threshold,
                "terminal_return": terminal_return,
                "max_forward_return": max_forward_return,
                "worst_forward_drawdown": worst_forward_drawdown,
                "terminal_recovery_margin": terminal_return - recovery_threshold,
                "anytime_recovery_margin": max_forward_return - recovery_threshold,
                "relapse_margin": relapse_threshold - worst_forward_drawdown,
                "anytime_outcome": anytime_label,
                "label_changes_under_anytime_recovery": anytime_label != ep.outcome,
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(output / "episode_label_audit.csv", index=False)

    mixed = audit[audit["current_outcome"] == "mixed"].copy()
    mixed.to_csv(output / "mixed_episode_audit.csv", index=False)

    subtype_counts = mixed["mixed_subtype"].value_counts(dropna=False).rename_axis("mixed_subtype").reset_index(name="count")
    subtype_counts.to_csv(output / "mixed_subtype_counts.csv", index=False)

    change_counts = (
        audit.groupby(["current_outcome", "anytime_outcome"])
        .size()
        .rename("count")
        .reset_index()
    )
    change_counts.to_csv(output / "terminal_vs_anytime_labels.csv", index=False)

    print(f"Data: {data_path}")
    print(f"Episodes: {len(audit)}")
    print("\nMixed subtype counts:")
    print(subtype_counts.to_string(index=False))
    print("\nTerminal-label -> anytime-recovery-label counts:")
    print(change_counts.to_string(index=False))
    print("\nMixed episodes ordered by proximity to a decisive threshold:")
    mixed = mixed.assign(
        closest_decisive_margin=np.maximum(
            mixed["anytime_recovery_margin"], mixed["relapse_margin"]
        )
    ).sort_values("closest_decisive_margin", ascending=False)
    print(
        mixed[
            [
                "branch_date",
                "mixed_subtype",
                "terminal_return",
                "max_forward_return",
                "worst_forward_drawdown",
                "recovery_threshold",
                "relapse_threshold",
                "anytime_outcome",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
