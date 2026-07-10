"""Run the final pre-freeze competing-escape tests.

The audit is intentionally narrow:

1. Force versus proximity: does downside pressure improve direction forecasts
   beyond current distance to the two first-passage barriers?
2. Timing coupling: does downside pressure improve total escape-hazard forecasts?
3. Clock shape: does a Weibull duration model beat a constant-rate exponential?

All predictive comparisons use episode-prequential training eligibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_competing_escape_tests import (
    CompetingEscapeTestConfig,
    add_first_passage_panel,
    prequential_clock_comparison,
    prequential_hazard_coupling,
    prequential_landmark_direction,
    summarize_clock_comparison,
    summarize_direction_predictions,
    summarize_hazard_predictions,
)
from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    IntermediateProbeConfig,
    build_first_passage_episode_table,
)

DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path(
    "results/diagnostics/branch_competing_escape_prefreeze_modern_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def model_metric(summary: pd.DataFrame, model: str, metric: str) -> float:
    row = summary[summary["model"] == model]
    return float(row.iloc[0][metric]) if len(row) else np.nan


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

    cfg = CompetingEscapeTestConfig()
    events = build_first_passage_episode_table(
        daily,
        episodes,
        IntermediateProbeConfig(max_followup=120),
    )
    events = events[events["resolved_within_followup"]].copy()
    panel = add_first_passage_panel(daily, events)

    # 1. Force versus proximity.
    direction_predictions = prequential_landmark_direction(panel, cfg)
    direction_summary = summarize_direction_predictions(direction_predictions)

    landmark_rows: list[dict[str, object]] = []
    for landmark in cfg.landmarks:
        subset = direction_summary[direction_summary["landmark_day"] == landmark]
        d1 = model_metric(subset, "D1_geometry", "logloss")
        d2 = model_metric(subset, "D2_geometry_plus_downside", "logloss")
        landmark_rows.append(
            {
                "landmark_day": landmark,
                "n_oos": int(subset["n_oos"].max()) if len(subset) else 0,
                "D1_geometry_logloss": d1,
                "D2_geometry_plus_downside_logloss": d2,
                "D2_minus_D1_logloss": d2 - d1,
                "downside_improves": bool(np.isfinite(d1) and np.isfinite(d2) and d2 < d1),
            }
        )
    force_vs_proximity = pd.DataFrame(landmark_rows)
    downside_landmarks_won = int(force_vs_proximity["downside_improves"].sum())

    # 2. Does downside pressure modulate total timing?
    hazard_scores = prequential_hazard_coupling(panel, cfg)
    hazard_summary = summarize_hazard_predictions(hazard_scores)
    h0_nll = model_metric(hazard_summary, "H0_constant", "mean_episode_nll")
    hp_nll = model_metric(hazard_summary, "HP_downside", "mean_episode_nll")
    h1_nll = model_metric(hazard_summary, "H1_duration", "mean_episode_nll")
    h1p_nll = model_metric(
        hazard_summary,
        "H1P_duration_plus_downside",
        "mean_episode_nll",
    )

    # 3. T0 versus T1 clock.
    clock_scores = prequential_clock_comparison(events, cfg)
    clock_summary = summarize_clock_comparison(clock_scores)
    t0_nll = model_metric(clock_summary, "T0_exponential", "mean_nll")
    t1_nll = model_metric(clock_summary, "T1_weibull", "mean_nll")

    verdicts = pd.DataFrame(
        [
            {
                "probe": "downside_pressure_survives_barrier_geometry",
                "question": "Does geometry+downside beat geometry-only logloss at at least 2 of 3 fixed landmarks?",
                "key_value": downside_landmarks_won,
                "verdict": "YES" if downside_landmarks_won >= 2 else "NO",
            },
            {
                "probe": "downside_pressure_modulates_total_escape_timing",
                "question": "Does downside-only improve mean OOS episode NLL versus constant hazard?",
                "key_value": hp_nll - h0_nll,
                "verdict": "YES" if np.isfinite(hp_nll) and np.isfinite(h0_nll) and hp_nll < h0_nll else "NO",
            },
            {
                "probe": "downside_adds_beyond_duration_for_timing",
                "question": "Does duration+downside improve mean OOS episode NLL versus duration-only?",
                "key_value": h1p_nll - h1_nll,
                "verdict": "YES" if np.isfinite(h1p_nll) and np.isfinite(h1_nll) and h1p_nll < h1_nll else "NO",
            },
            {
                "probe": "duration_dependent_clock_beats_constant_rate",
                "question": "Does Weibull T1 improve mean OOS NLL versus exponential T0?",
                "key_value": t1_nll - t0_nll,
                "verdict": "YES" if np.isfinite(t1_nll) and np.isfinite(t0_nll) and t1_nll < t0_nll else "NO",
            },
        ]
    )

    events.to_csv(args.output / "first_passage_episodes.csv", index=False)
    panel.to_csv(args.output / "first_passage_risk_panel.csv", index=False)
    direction_predictions.to_csv(args.output / "direction_prequential_predictions.csv", index=False)
    direction_summary.to_csv(args.output / "direction_summary.csv", index=False)
    force_vs_proximity.to_csv(args.output / "force_vs_proximity_landmarks.csv", index=False)
    hazard_scores.to_csv(args.output / "hazard_prequential_episode_scores.csv", index=False)
    hazard_summary.to_csv(args.output / "hazard_summary.csv", index=False)
    clock_scores.to_csv(args.output / "clock_prequential_scores.csv", index=False)
    clock_summary.to_csv(args.output / "clock_summary.csv", index=False)
    verdicts.to_csv(args.output / "probe_verdicts.csv", index=False)

    manifest = {
        "status": "final pre-freeze tests",
        "independent_episode_count": int(events["episode_id"].nunique()),
        "small_sample_warning": (
            "The risk panel creates repeated at-risk rows but does not increase the number of independent crisis episodes. "
            "Predictive scores are aggregated by held-out episode; small independent n remains unresolved."
        ),
        "direction_landmarks": list(cfg.landmarks),
        "direction_models": {
            "D0": "historical recovery rate",
            "D1": "running distance to recovery and relapse barriers",
            "D2": "D1 plus downside pressure",
        },
        "timing_models": {
            "H0": "constant daily hazard estimated from prior completed episodes",
            "H1": "log duration",
            "HP": "downside pressure",
            "H1P": "log duration plus downside pressure",
        },
        "clock_models": {
            "T0": "exponential",
            "T1": "Weibull",
        },
        "verdicts_are_not_significance_tests": True,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Resolved independent episodes: {events['episode_id'].nunique()}")
    print(f"At-risk panel rows: {len(panel)}")
    print("WARNING: panel rows are repeated observations; independent n remains episode count.\n")

    print("Final pre-freeze probes:")
    print(verdicts.to_string(index=False))
    print("\nForce versus proximity by landmark:")
    print(force_vs_proximity.to_string(index=False))
    print("\nDirection model summary:")
    print(direction_summary.to_string(index=False))
    print("\nTiming hazard summary:")
    print(hazard_summary.to_string(index=False))
    print("\nClock comparison:")
    print(clock_summary.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
