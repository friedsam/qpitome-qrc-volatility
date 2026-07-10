"""Simple falsification audit for competing interpretations of ``mixed`` episodes.

The current 40-day branch label calls an episode ``mixed`` when either both
current recovery/relapse conditions are met or neither is met. This diagnostic
does not fit a model. It asks four deliberately simple questions with fixed,
pre-specified operational verdicts:

1. LABEL-RULE ARTIFACT:
   Did a mixed episode already cross one of the original fixed barriers first
   within 40 days? If so, a first-passage formulation would have resolved it.

2. THRESHOLD ARTIFACT:
   Among episodes with no original-barrier crossing by day 40, does moving both
   barriers 20% closer resolve many of them within the same 40-day window?

3. 40-DAY CENSORING:
   Among episodes with no original-barrier crossing by day 40, do most cross an
   original fixed barrier between days 41 and 120?

4. PERSISTENT METASTABLE / UNRESOLVED REGION:
   Among episodes with no original-barrier crossing by day 40, does a substantial
   fraction remain unescaped through day 120 while spending at least half of the
   post-40 observation period in a high-stress, persistently damaged core state?

These interpretations are not mutually exclusive. The YES/NO thresholds are
operational screening rules, not hypothesis-test p-values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branching_state import (
    BranchStateConfig,
    OutcomeConfig,
    detect_branch_state,
)

DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path("results/diagnostics/mixed_outcome_interpretations_modern_v1")

PRIMARY_HORIZON = 40
EXTENDED_HORIZON = 120
REPORT_HORIZONS = (40, 60, 80, 120, 180, 252)
CLOSER_BARRIER_FACTOR = 0.80

# Operational verdict thresholds. These are intentionally simple and fixed.
LABEL_RULE_YES_FRACTION = 1.0 / 3.0
THRESHOLD_ARTIFACT_YES_FRACTION = 1.0 / 3.0
CENSORING_YES_FRACTION = 0.50
PERSISTENT_YES_FRACTION = 1.0 / 3.0
PERSISTENT_CORE_OCCUPANCY = 0.50
MIN_DENOMINATOR_FOR_VERDICT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def first_passage_day(
    running_return: np.ndarray,
    upper: float,
    lower: float,
) -> tuple[int | None, str | None]:
    """Return 1-based first crossing day and barrier name, or (None, None)."""
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
    # A one-dimensional return cannot be above a positive barrier and below a
    # negative barrier on the same row, but keep this explicit for safety.
    return upper_day, "simultaneous"


def verdict(value: bool, denominator: int) -> str:
    if denominator < MIN_DENOMINATOR_FOR_VERDICT:
        return "INCONCLUSIVE"
    return "YES" if value else "NO"


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def main() -> None:
    args = parse_args()
    for path in (args.data, args.episodes):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)

    daily = (
        pd.read_csv(args.data, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    episodes = pd.read_csv(args.episodes, parse_dates=["branch_date"])
    required_episode = {
        "episode_id",
        "branch_idx",
        "outcome",
        "outcome_complete",
        "rv_20d",
    }
    missing = required_episode - set(episodes.columns)
    if missing:
        raise KeyError(f"Missing episode columns: {sorted(missing)}")

    detected = detect_branch_state(daily, BranchStateConfig())
    prices = detected["spy_adj_close"].to_numpy(dtype=float)
    core_state = (
        detected["branch_high_stress"].fillna(False).to_numpy(dtype=bool)
        & detected["branch_persistent_damage"].fillna(False).to_numpy(dtype=bool)
    )

    outcome_cfg = OutcomeConfig()
    mixed = episodes[
        episodes["outcome_complete"].astype(bool) & (episodes["outcome"] == "mixed")
    ].copy()
    if mixed.empty:
        raise RuntimeError("No complete mixed episodes found")

    rows: list[dict[str, object]] = []
    for episode in mixed.itertuples(index=False):
        branch_idx = int(episode.branch_idx)
        max_available = min(max(REPORT_HORIZONS), len(prices) - branch_idx - 1)
        if max_available < PRIMARY_HORIZON:
            continue

        branch_price = prices[branch_idx]
        future_prices = prices[branch_idx + 1 : branch_idx + max_available + 1]
        running_return = future_prices / branch_price - 1.0

        scale = float(episode.rv_20d) * np.sqrt(PRIMARY_HORIZON / 252.0)
        upper = outcome_cfg.recovery_scale * scale
        lower = -outcome_cfg.relapse_scale * scale
        closer_upper = CLOSER_BARRIER_FACTOR * upper
        closer_lower = CLOSER_BARRIER_FACTOR * lower

        fp_day, fp_type = first_passage_day(running_return, upper, lower)
        close_day, close_type = first_passage_day(
            running_return[:PRIMARY_HORIZON], closer_upper, closer_lower
        )

        fixed_unresolved_40 = fp_day is None or fp_day > PRIMARY_HORIZON
        resolved_by_40_first_passage = fp_day is not None and fp_day <= PRIMARY_HORIZON
        late_resolved_41_120 = (
            fp_day is not None and PRIMARY_HORIZON < fp_day <= EXTENDED_HORIZON
        )
        unresolved_120 = fp_day is None or fp_day > EXTENDED_HORIZON
        threshold_resolved_40 = (
            fixed_unresolved_40 and close_day is not None and close_day <= PRIMARY_HORIZON
        )

        post40_end = min(EXTENDED_HORIZON, max_available)
        if post40_end > PRIMARY_HORIZON:
            core_slice = core_state[
                branch_idx + PRIMARY_HORIZON + 1 : branch_idx + post40_end + 1
            ]
            core_occupancy_41_120 = float(np.mean(core_slice)) if len(core_slice) else np.nan
        else:
            core_occupancy_41_120 = np.nan

        row: dict[str, object] = {
            "episode_id": int(episode.episode_id),
            "branch_date": episode.branch_date,
            "current_outcome": episode.outcome,
            "original_upper_barrier": upper,
            "original_lower_barrier": lower,
            "fixed_first_passage_day": fp_day,
            "fixed_first_passage_type": fp_type,
            "resolved_by_40_first_passage": resolved_by_40_first_passage,
            "fixed_unresolved_40": fixed_unresolved_40,
            "late_resolved_41_120": late_resolved_41_120,
            "fixed_unresolved_120": unresolved_120,
            "closer_first_passage_day_40": close_day,
            "closer_first_passage_type_40": close_type,
            "threshold_resolved_40": threshold_resolved_40,
            "core_stress_occupancy_days_41_120": core_occupancy_41_120,
            "persistent_metastable_candidate": (
                fixed_unresolved_40
                and unresolved_120
                and np.isfinite(core_occupancy_41_120)
                and core_occupancy_41_120 >= PERSISTENT_CORE_OCCUPANCY
            ),
        }
        for horizon in REPORT_HORIZONS:
            row[f"resolved_by_{horizon}"] = bool(fp_day is not None and fp_day <= horizon)
        rows.append(row)

    detail = pd.DataFrame(rows)
    n_mixed = len(detail)
    unresolved40 = detail[detail["fixed_unresolved_40"]]
    n_unresolved40 = len(unresolved40)

    n_rule = int(detail["resolved_by_40_first_passage"].sum())
    n_threshold = int(unresolved40["threshold_resolved_40"].sum())
    n_late = int(unresolved40["late_resolved_41_120"].sum())
    n_persistent = int(unresolved40["persistent_metastable_candidate"].sum())

    rule_fraction = safe_fraction(n_rule, n_mixed)
    threshold_fraction = safe_fraction(n_threshold, n_unresolved40)
    censor_fraction = safe_fraction(n_late, n_unresolved40)
    persistent_fraction = safe_fraction(n_persistent, n_unresolved40)

    verdict_rows = [
        {
            "interpretation": "label_rule_artifact",
            "question": "Do >=1/3 of mixed episodes already resolve by first passage within 40 days?",
            "numerator": n_rule,
            "denominator": n_mixed,
            "fraction": rule_fraction,
            "threshold": LABEL_RULE_YES_FRACTION,
            "verdict": verdict(
                bool(rule_fraction >= LABEL_RULE_YES_FRACTION), n_mixed
            ),
        },
        {
            "interpretation": "threshold_artifact",
            "question": "Among original-barrier unresolved episodes, do >=1/3 resolve within 40 days when both barriers move 20% closer?",
            "numerator": n_threshold,
            "denominator": n_unresolved40,
            "fraction": threshold_fraction,
            "threshold": THRESHOLD_ARTIFACT_YES_FRACTION,
            "verdict": verdict(
                bool(threshold_fraction >= THRESHOLD_ARTIFACT_YES_FRACTION),
                n_unresolved40,
            ),
        },
        {
            "interpretation": "forty_day_censoring",
            "question": "Among original-barrier unresolved episodes, do >=1/2 first escape during days 41-120?",
            "numerator": n_late,
            "denominator": n_unresolved40,
            "fraction": censor_fraction,
            "threshold": CENSORING_YES_FRACTION,
            "verdict": verdict(
                bool(censor_fraction >= CENSORING_YES_FRACTION), n_unresolved40
            ),
        },
        {
            "interpretation": "persistent_metastable_region",
            "question": "Among original-barrier unresolved episodes, do >=1/3 remain unescaped through 120 days and spend >=1/2 of days 41-120 in high-stress persistent damage?",
            "numerator": n_persistent,
            "denominator": n_unresolved40,
            "fraction": persistent_fraction,
            "threshold": PERSISTENT_YES_FRACTION,
            "verdict": verdict(
                bool(persistent_fraction >= PERSISTENT_YES_FRACTION),
                n_unresolved40,
            ),
        },
    ]
    verdicts = pd.DataFrame(verdict_rows)

    resolution_rows = []
    for horizon in REPORT_HORIZONS:
        resolved = int(detail[f"resolved_by_{horizon}"].sum())
        resolution_rows.append(
            {
                "horizon": horizon,
                "resolved": resolved,
                "mixed_total": n_mixed,
                "resolved_fraction": safe_fraction(resolved, n_mixed),
            }
        )
    resolution_curve = pd.DataFrame(resolution_rows)

    # Exact mutually exclusive decomposition under fixed original barriers.
    decomposition = pd.DataFrame(
        [
            {
                "bucket": "resolved_by_first_passage_within_40",
                "n": n_rule,
                "fraction_of_mixed": safe_fraction(n_rule, n_mixed),
            },
            {
                "bucket": "first_passage_days_41_120",
                "n": n_late,
                "fraction_of_mixed": safe_fraction(n_late, n_mixed),
            },
            {
                "bucket": "still_unresolved_at_120",
                "n": int(unresolved40["fixed_unresolved_120"].sum()),
                "fraction_of_mixed": safe_fraction(
                    int(unresolved40["fixed_unresolved_120"].sum()), n_mixed
                ),
            },
        ]
    )

    detail.to_csv(args.output / "mixed_episode_detail.csv", index=False)
    verdicts.to_csv(args.output / "interpretation_verdicts.csv", index=False)
    resolution_curve.to_csv(args.output / "fixed_barrier_resolution_curve.csv", index=False)
    decomposition.to_csv(args.output / "mixed_first_passage_decomposition.csv", index=False)

    manifest = {
        "status": "descriptive falsification audit; not a fitted model",
        "current_label_rule": (
            "recovery uses terminal 40-day return; relapse uses worst 40-day drawdown; "
            "both-hit and neither-hit cases are mixed"
        ),
        "fixed_barrier_first_passage_test": {
            "barriers": "original 40-day volatility-scaled thresholds held fixed at all follow-up horizons",
            "primary_horizon": PRIMARY_HORIZON,
            "extended_horizon": EXTENDED_HORIZON,
        },
        "closer_barrier_sensitivity": {
            "factor": CLOSER_BARRIER_FACTOR,
            "meaning": "both original barrier magnitudes multiplied by 0.8",
        },
        "persistent_core_state": (
            "branch_high_stress AND branch_persistent_damage from the causal detector"
        ),
        "verdict_thresholds": {
            "label_rule_artifact": LABEL_RULE_YES_FRACTION,
            "threshold_artifact": THRESHOLD_ARTIFACT_YES_FRACTION,
            "forty_day_censoring": CENSORING_YES_FRACTION,
            "persistent_metastable_region": PERSISTENT_YES_FRACTION,
            "persistent_core_occupancy": PERSISTENT_CORE_OCCUPANCY,
            "minimum_denominator": MIN_DENOMINATOR_FOR_VERDICT,
        },
        "warning": "YES/NO thresholds are operational screens, not statistical significance tests; interpretations can overlap.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Complete mixed episodes: {n_mixed}")
    print(f"Original-barrier unresolved at day 40: {n_unresolved40}")
    print("\nSimple interpretation verdicts:")
    print(
        verdicts[
            ["interpretation", "numerator", "denominator", "fraction", "verdict"]
        ].to_string(index=False)
    )
    print("\nFixed-barrier first-passage decomposition:")
    print(decomposition.to_string(index=False))
    print("\nFixed-barrier resolution curve:")
    print(resolution_curve.to_string(index=False))
    print("\nPer-episode detail:")
    print(
        detail[
            [
                "episode_id",
                "branch_date",
                "fixed_first_passage_day",
                "fixed_first_passage_type",
                "threshold_resolved_40",
                "fixed_unresolved_120",
                "core_stress_occupancy_days_41_120",
                "persistent_metastable_candidate",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
