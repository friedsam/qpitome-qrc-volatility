"""Operational audit of slower internal relaxation around branch-state entry.

The runner answers four deliberately narrow questions:

1. Does persistence increase after entry?
2. Do downside-shock responses decay more slowly after entry?
3. Do deviations last longer after entry?
4. Does restoring behavior recover near escape?

All comparisons are descriptive and mostly episode-paired. The Y/N verdicts are
fixed admission screens, not significance tests. The detailed tables are saved
for interpretation, especially when episodes have heterogeneous timescales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_relaxation_dynamics import (
    RELAXATION_VARIABLES,
    RelaxationProbeConfig,
    build_relaxation_segment_table,
)

DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path("results/diagnostics/branch_relaxation_dynamics_modern_v1")

MIN_PAIRS = 8
PERSISTENCE_REQUIRED_VARIABLES = 2
PERSISTENCE_MIN_MEDIAN_DELTA = 0.05
SHOCK_SLOWING_MIN_RATIO = 1.10
RUN_LENGTH_REQUIRED_VARIABLES = 2
RUN_LENGTH_MIN_RATIO = 1.10
ESCAPE_RECOVERY_REQUIRED_VARIABLES = 2
RESTORING_MIN_MEDIAN_DELTA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def verdict(condition: bool, n: int) -> str:
    if n < MIN_PAIRS:
        return "INCONCLUSIVE"
    return "YES" if condition else "NO"


def paired_comparison(
    metrics: pd.DataFrame,
    left_segment: str,
    right_segment: str,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        pivot = metrics.pivot(index="episode_id", columns="segment", values=column)
        if not {left_segment, right_segment}.issubset(pivot.columns):
            continue
        pair = pivot[[left_segment, right_segment]].dropna()
        for episode_id, values in pair.iterrows():
            left = float(values[left_segment])
            right = float(values[right_segment])
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "metric": column,
                    "left_segment": left_segment,
                    "right_segment": right_segment,
                    "left_value": left,
                    "right_value": right,
                    "difference": right - left,
                    "ratio": right / left if left != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, group in pairs.groupby("metric"):
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(group),
                "median_left": float(group["left_value"].median()),
                "median_right": float(group["right_value"].median()),
                "median_difference": float(group["difference"].median()),
                "median_ratio": float(group["ratio"].replace([np.inf, -np.inf], np.nan).median()),
                "fraction_difference_positive": float((group["difference"] > 0).mean()),
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

    cfg = RelaxationProbeConfig()
    events, metrics = build_relaxation_segment_table(daily, episodes, cfg)

    lag1_columns = [f"{v}__lag1" for v in RELAXATION_VARIABLES]
    run_columns = [f"{v}__same_side_run_length" for v in RELAXATION_VARIABLES]
    restoring_columns = [f"{v}__restoring_fraction" for v in RELAXATION_VARIABLES]
    shock_columns = ["log_rv_ratio__shock_carryover_lag5"]

    pre_to_early = paired_comparison(
        metrics,
        "pre_entry",
        "early_residence",
        lag1_columns + run_columns + shock_columns,
    )
    early_to_late = paired_comparison(
        metrics,
        "early_residence",
        "late_residence",
        lag1_columns + restoring_columns,
    )
    pre_to_early_summary = summarize_pairs(pre_to_early)
    early_to_late_summary = summarize_pairs(early_to_late)

    persistence_rows = pre_to_early_summary[
        pre_to_early_summary["metric"].isin(lag1_columns)
    ].copy()
    persistence_positive = persistence_rows[
        persistence_rows["median_difference"] >= PERSISTENCE_MIN_MEDIAN_DELTA
    ]
    persistence_n = int(persistence_rows["n_pairs"].max()) if len(persistence_rows) else 0

    shock_row = pre_to_early_summary[
        pre_to_early_summary["metric"] == shock_columns[0]
    ]
    shock_ratio = float(shock_row["median_ratio"].iloc[0]) if len(shock_row) else np.nan
    shock_n = int(shock_row["n_pairs"].iloc[0]) if len(shock_row) else 0

    run_rows = pre_to_early_summary[
        pre_to_early_summary["metric"].isin(run_columns)
    ].copy()
    run_positive = run_rows[run_rows["median_ratio"] >= RUN_LENGTH_MIN_RATIO]
    run_n = int(run_rows["n_pairs"].max()) if len(run_rows) else 0

    late_lag = early_to_late_summary[
        early_to_late_summary["metric"].isin(lag1_columns)
    ].copy()
    late_restore = early_to_late_summary[
        early_to_late_summary["metric"].isin(restoring_columns)
    ].copy()
    persistence_recovers = int((late_lag["median_difference"] <= -PERSISTENCE_MIN_MEDIAN_DELTA).sum())
    restoring_recovers = int((late_restore["median_difference"] >= RESTORING_MIN_MEDIAN_DELTA).sum())
    escape_recovery_score = persistence_recovers + restoring_recovers
    escape_n = int(
        max(
            late_lag["n_pairs"].max() if len(late_lag) else 0,
            late_restore["n_pairs"].max() if len(late_restore) else 0,
        )
    )

    verdicts = pd.DataFrame(
        [
            {
                "probe": "persistence_increases_after_entry",
                "question": "Do at least 2 of 3 relaxation variables increase lag-1 persistence after entry by median delta >=0.05?",
                "key_value": int(len(persistence_positive)),
                "verdict": verdict(
                    len(persistence_positive) >= PERSISTENCE_REQUIRED_VARIABLES,
                    persistence_n,
                ),
            },
            {
                "probe": "shock_recovery_slows_after_entry",
                "question": "Is median lag-5 volatility-response carryover at least 10% larger after entry?",
                "key_value": shock_ratio,
                "verdict": verdict(
                    bool(np.isfinite(shock_ratio) and shock_ratio >= SHOCK_SLOWING_MIN_RATIO),
                    shock_n,
                ),
            },
            {
                "probe": "deviations_last_longer_after_entry",
                "question": "Do at least 2 of 3 variables show >=10% longer same-side runs after entry?",
                "key_value": int(len(run_positive)),
                "verdict": verdict(
                    len(run_positive) >= RUN_LENGTH_REQUIRED_VARIABLES,
                    run_n,
                ),
            },
            {
                "probe": "relaxation_recovers_near_escape",
                "question": "Do at least 2 persistence/restoring indicators move toward faster restoration near escape?",
                "key_value": escape_recovery_score,
                "verdict": verdict(
                    escape_recovery_score >= ESCAPE_RECOVERY_REQUIRED_VARIABLES,
                    escape_n,
                ),
            },
        ]
    )

    events.to_csv(args.output / "first_passage_episodes.csv", index=False)
    metrics.to_csv(args.output / "segment_relaxation_metrics.csv", index=False)
    pre_to_early.to_csv(args.output / "paired_pre_to_early_relaxation.csv", index=False)
    early_to_late.to_csv(args.output / "paired_early_to_late_relaxation.csv", index=False)
    pre_to_early_summary.to_csv(args.output / "pre_to_early_summary.csv", index=False)
    early_to_late_summary.to_csv(args.output / "early_to_late_summary.csv", index=False)
    verdicts.to_csv(args.output / "probe_verdicts.csv", index=False)

    manifest = {
        "status": "model-free relaxation audit",
        "hypothesis": "branch episodes may become observable because internal relaxation slows relative to fixed daily sampling",
        "interpretation_warning": (
            "episode-specific timescales and heterogeneous intermediates can weaken pooled verdicts; "
            "paired episode-level tables remain primary evidence"
        ),
        "segments": {
            "pre_entry_days": cfg.pre_window,
            "early_residence_max_days": cfg.early_window,
            "late_residence_max_days": cfg.late_window,
            "escape_day_excluded": True,
        },
        "verdicts_are_not_p_values": True,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Complete episodes supplied: {len(episodes)}")
    print(f"Resolved by fixed-barrier first passage within 120 days: {len(events)}")
    print("\nRelaxation probes:")
    print(verdicts.to_string(index=False))
    print("\nPre-entry -> early-residence summary:")
    print(pre_to_early_summary.to_string(index=False))
    print("\nEarly-residence -> late-residence summary:")
    print(early_to_late_summary.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
