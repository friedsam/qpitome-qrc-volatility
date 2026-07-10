"""Model-free probes of the branch state as a transient intermediate.

This runner stays upstream of hazard, HMM, Bell-Kramers, or QRC modeling. It
reconstructs fixed-barrier first-passage events for all complete branch episodes
and asks a small set of operational YES/NO questions:

1. OBSERVABLE RESIDENCE: is the branch state usually occupied long enough to be
   more than an instantaneous boundary crossing?
2. FLUCTUATING RESIDENCE: are early post-entry trajectories more directionally
   alternating/choppy than the matched pre-entry path?
3. COMMON EARLY INTERMEDIATE: are recovery- and relapse-bound episodes weakly
   separated during the first part of residence?
4. LATE DIVERGENCE: do recovery- and relapse-bound paths separate more strongly
   near first passage than early in residence?
5. CONSTANT ESCAPE RATE: is a memoryless total escape-rate approximation roughly
   compatible with the empirical duration profile?

The thresholds are descriptive admission screens, not p-values. The detailed
metric tables are always saved so a borderline verdict can be inspected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    DYNAMIC_METRICS,
    IntermediateProbeConfig,
    build_first_passage_episode_table,
    build_intermediate_segment_metrics,
    standardized_mean_difference,
)

DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path("results/diagnostics/branch_intermediate_dynamics_modern_v1")

# Fixed operational screens.
RESIDENCE_DAY = 10
RESIDENCE_YES_FRACTION = 0.50
FLUCTUATION_SWITCH_DELTA = 0.05
FLUCTUATION_COHERENCE_DELTA = -0.05
FLUCTUATION_CHOPPINESS_RATIO = 1.10
FLUCTUATION_REQUIRED_INDICATORS = 2
EARLY_COMMON_MAX_ABS_SMD = 0.50
LATE_DIVERGENCE_MAX_ABS_SMD = 0.80
LATE_VS_EARLY_MEDIAN_RATIO = 1.50
CONSTANT_RATE_MAX_MIN_RATIO = 2.0
MIN_VERDICT_EPISODES = 6
DURATION_BINS = ((1, 20), (21, 40), (41, 80), (81, 120))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def verdict(condition: bool, n: int) -> str:
    if n < MIN_VERDICT_EPISODES:
        return "INCONCLUSIVE"
    return "YES" if condition else "NO"


def paired_segment_changes(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in DYNAMIC_METRICS:
        pivot = metrics.pivot(index="episode_id", columns="segment", values=metric)
        if not {"pre_entry", "early_residence"}.issubset(pivot.columns):
            continue
        pair = pivot[["pre_entry", "early_residence"]].dropna()
        for episode_id, values in pair.iterrows():
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "metric": metric,
                    "pre_entry": float(values["pre_entry"]),
                    "early_residence": float(values["early_residence"]),
                    "difference": float(values["early_residence"] - values["pre_entry"]),
                    "ratio": (
                        float(values["early_residence"] / values["pre_entry"])
                        if values["pre_entry"] != 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def outcome_separation(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment in ("early_residence", "late_residence"):
        subset = metrics[metrics["segment"] == segment]
        for metric in DYNAMIC_METRICS:
            recovery = subset.loc[subset["event_type"] == "recovery", metric].to_numpy(dtype=float)
            relapse = subset.loc[subset["event_type"] == "relapse", metric].to_numpy(dtype=float)
            rows.append(
                {
                    "segment": segment,
                    "metric": metric,
                    "n_recovery": int(np.isfinite(recovery).sum()),
                    "n_relapse": int(np.isfinite(relapse).sum()),
                    "smd_recovery_minus_relapse": standardized_mean_difference(recovery, relapse),
                }
            )
    return pd.DataFrame(rows)


def duration_hazard_table(events: pd.DataFrame) -> pd.DataFrame:
    resolved = events[events["resolved_within_followup"]].copy()
    rows: list[dict[str, object]] = []
    for start, end in DURATION_BINS:
        at_risk = int((resolved["event_day"] >= start).sum())
        escaped = int(((resolved["event_day"] >= start) & (resolved["event_day"] <= end)).sum())
        width = end - start + 1
        bin_probability = escaped / at_risk if at_risk else np.nan
        if at_risk and bin_probability < 1.0:
            daily_rate = -np.log(1.0 - bin_probability) / width
        elif at_risk and bin_probability == 1.0:
            daily_rate = np.inf
        else:
            daily_rate = np.nan
        rows.append(
            {
                "start_day": start,
                "end_day": end,
                "width": width,
                "at_risk": at_risk,
                "escaped": escaped,
                "escape_probability_in_bin": bin_probability,
                "implied_daily_escape_rate": daily_rate,
            }
        )
    return pd.DataFrame(rows)


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
    if "outcome_complete" in episodes.columns:
        episodes = episodes[episodes["outcome_complete"].astype(bool)].copy()

    cfg = IntermediateProbeConfig()
    events = build_first_passage_episode_table(daily, episodes, cfg)
    metrics = build_intermediate_segment_metrics(daily, events, cfg)
    changes = paired_segment_changes(metrics)
    separation = outcome_separation(metrics)
    hazard = duration_hazard_table(events)

    # Probe 1: nontrivial residence.
    resolved = events[events["resolved_within_followup"]]
    n_resolved = len(resolved)
    survive_10 = int((resolved["event_day"] > RESIDENCE_DAY).sum())
    survive_10_fraction = survive_10 / n_resolved if n_resolved else np.nan

    # Probe 2: increased fluctuation after entry.
    fluctuation_checks: list[dict[str, object]] = []
    indicator_specs = (
        ("return_sign_switch_rate", "difference", lambda x: x >= FLUCTUATION_SWITCH_DELTA),
        ("repair_sign_switch_rate", "difference", lambda x: x >= FLUCTUATION_SWITCH_DELTA),
        ("directional_coherence", "difference", lambda x: x <= FLUCTUATION_COHERENCE_DELTA),
        ("rv_ratio_choppiness", "ratio", lambda x: x >= FLUCTUATION_CHOPPINESS_RATIO),
    )
    for metric, value_column, condition in indicator_specs:
        group = changes[changes["metric"] == metric]
        value = float(group[value_column].median()) if len(group) else np.nan
        fluctuation_checks.append(
            {
                "metric": metric,
                "n_pairs": len(group),
                "summary_kind": f"median_{value_column}",
                "summary_value": value,
                "supports_more_fluctuation": bool(np.isfinite(value) and condition(value)),
            }
        )
    fluctuation_table = pd.DataFrame(fluctuation_checks)
    n_fluctuation_positive = int(fluctuation_table["supports_more_fluctuation"].sum())
    fluctuation_n = int(fluctuation_table["n_pairs"].max()) if len(fluctuation_table) else 0

    # Probe 3 and 4: early commonality and late divergence by outcome.
    early = separation[separation["segment"] == "early_residence"].copy()
    late = separation[separation["segment"] == "late_residence"].copy()
    early_abs = np.abs(early["smd_recovery_minus_relapse"].to_numpy(dtype=float))
    late_abs = np.abs(late["smd_recovery_minus_relapse"].to_numpy(dtype=float))
    early_abs = early_abs[np.isfinite(early_abs)]
    late_abs = late_abs[np.isfinite(late_abs)]
    early_max = float(np.max(early_abs)) if len(early_abs) else np.nan
    early_median = float(np.median(early_abs)) if len(early_abs) else np.nan
    late_max = float(np.max(late_abs)) if len(late_abs) else np.nan
    late_median = float(np.median(late_abs)) if len(late_abs) else np.nan
    divergence_ratio = (
        late_median / early_median
        if np.isfinite(late_median) and np.isfinite(early_median) and early_median > 0
        else np.nan
    )
    outcome_n = int(
        min(
            early["n_recovery"].min() if len(early) else 0,
            early["n_relapse"].min() if len(early) else 0,
        )
    )

    # Probe 5: rough memoryless-rate compatibility.
    finite_rates = hazard.loc[
        (hazard["at_risk"] >= MIN_VERDICT_EPISODES)
        & np.isfinite(hazard["implied_daily_escape_rate"]),
        "implied_daily_escape_rate",
    ].to_numpy(dtype=float)
    rate_ratio = (
        float(np.max(finite_rates) / np.min(finite_rates))
        if len(finite_rates) >= 2 and np.min(finite_rates) > 0
        else np.nan
    )

    verdicts = pd.DataFrame(
        [
            {
                "probe": "observable_residence",
                "question": f"Do at least half of resolved episodes survive more than {RESIDENCE_DAY} days after branch entry?",
                "key_value": survive_10_fraction,
                "verdict": verdict(
                    bool(np.isfinite(survive_10_fraction) and survive_10_fraction >= RESIDENCE_YES_FRACTION),
                    n_resolved,
                ),
            },
            {
                "probe": "fluctuating_residence",
                "question": "Do at least 2 of 4 directional/choppiness indicators increase from pre-entry to early residence?",
                "key_value": n_fluctuation_positive,
                "verdict": verdict(
                    n_fluctuation_positive >= FLUCTUATION_REQUIRED_INDICATORS,
                    fluctuation_n,
                ),
            },
            {
                "probe": "common_early_intermediate",
                "question": "Is the maximum early recovery-vs-relapse dynamic separation below |SMD|=0.5?",
                "key_value": early_max,
                "verdict": verdict(
                    bool(np.isfinite(early_max) and early_max < EARLY_COMMON_MAX_ABS_SMD),
                    outcome_n,
                ),
            },
            {
                "probe": "late_outcome_divergence",
                "question": "Does late separation reach |SMD|>=0.8 and median separation grow >=1.5x versus early residence?",
                "key_value": divergence_ratio,
                "verdict": verdict(
                    bool(
                        np.isfinite(late_max)
                        and np.isfinite(divergence_ratio)
                        and late_max >= LATE_DIVERGENCE_MAX_ABS_SMD
                        and divergence_ratio >= LATE_VS_EARLY_MEDIAN_RATIO
                    ),
                    outcome_n,
                ),
            },
            {
                "probe": "constant_escape_rate_roughly_compatible",
                "question": "Across duration bins with adequate risk sets, is max/min implied daily escape rate <=2?",
                "key_value": rate_ratio,
                "verdict": verdict(
                    bool(np.isfinite(rate_ratio) and rate_ratio <= CONSTANT_RATE_MAX_MIN_RATIO),
                    n_resolved,
                ),
            },
        ]
    )

    events.to_csv(args.output / "first_passage_episodes.csv", index=False)
    metrics.to_csv(args.output / "segment_dynamics.csv", index=False)
    changes.to_csv(args.output / "paired_pre_to_early_changes.csv", index=False)
    fluctuation_table.to_csv(args.output / "fluctuation_indicator_summary.csv", index=False)
    separation.to_csv(args.output / "outcome_separation_by_segment.csv", index=False)
    hazard.to_csv(args.output / "duration_escape_rate_bins.csv", index=False)
    verdicts.to_csv(args.output / "probe_verdicts.csv", index=False)

    manifest = {
        "status": "model-free descriptive probes before hazard/HMM/Bell-Kramers modeling",
        "first_passage": "original 40-day volatility-scaled barriers held fixed through 120-day follow-up",
        "segments": {
            "pre_entry": cfg.pre_window,
            "early_residence_max": cfg.early_window,
            "late_residence_max": cfg.late_window,
            "escape_day_excluded_from dynamics": True,
        },
        "warning": "YES/NO verdicts are operational screens, not hypothesis-test p-values",
        "thresholds": {
            "observable_residence_day": RESIDENCE_DAY,
            "observable_residence_fraction": RESIDENCE_YES_FRACTION,
            "fluctuation_switch_delta": FLUCTUATION_SWITCH_DELTA,
            "fluctuation_coherence_delta": FLUCTUATION_COHERENCE_DELTA,
            "fluctuation_choppiness_ratio": FLUCTUATION_CHOPPINESS_RATIO,
            "common_early_max_abs_smd": EARLY_COMMON_MAX_ABS_SMD,
            "late_divergence_max_abs_smd": LATE_DIVERGENCE_MAX_ABS_SMD,
            "late_vs_early_median_ratio": LATE_VS_EARLY_MEDIAN_RATIO,
            "constant_rate_max_min_ratio": CONSTANT_RATE_MAX_MIN_RATIO,
        },
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Complete episodes: {len(events)}")
    print(f"Resolved by fixed-barrier first passage within 120 days: {n_resolved}")
    print("\nSimple intermediate-state probes:")
    print(verdicts.to_string(index=False))
    print("\nFluctuation indicators:")
    print(fluctuation_table.to_string(index=False))
    print("\nRecovery-vs-relapse separation by segment:")
    print(separation.to_string(index=False))
    print("\nDuration-binned total escape rates:")
    print(hazard.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
