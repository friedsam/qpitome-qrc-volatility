"""Run the final three descriptive probes before target-design freeze.

1. Residence time versus eventual escape direction.
2. Branch-state relaxation versus matched high-stress non-branch controls.
3. Resolution-aligned recovery-versus-relapse commitment window.

The outputs are descriptive. Y/N screens are operational and are not p-values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_escape_diagnostics import (
    COMMITMENT_FEATURES,
    EscapeDiagnosticConfig,
    build_matched_control_relaxation,
    build_resolution_aligned_separation,
    commitment_window,
    match_high_stress_controls,
    rank_biserial_recovery_minus_relapse,
    residence_direction_summary,
)
from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    IntermediateProbeConfig,
    build_first_passage_episode_table,
)

DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path("results/diagnostics/branch_escape_structure_modern_v1")

MIN_GROUP_N = 8
DIRECTION_EFFECT_THRESHOLD = 0.33
CONTROL_LAG1_DELTA = 0.05
CONTROL_SHOCK_RATIO = 1.25
CONTROL_REQUIRED_INDICATORS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def verdict(condition: bool, n: int) -> str:
    if n < MIN_GROUP_N:
        return "INCONCLUSIVE"
    return "YES" if condition else "NO"


def paired_control_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = [
        column
        for column in metrics.columns
        if column not in {"episode_id", "kind", "anchor_idx"}
    ]
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for metric in value_columns:
        pivot = metrics.pivot(index="episode_id", columns="kind", values=metric)
        if not {"branch", "control"}.issubset(pivot.columns):
            continue
        pair = pivot[["control", "branch"]].dropna()
        for episode_id, values in pair.iterrows():
            control = float(values["control"])
            branch = float(values["branch"])
            detail_rows.append(
                {
                    "episode_id": int(episode_id),
                    "metric": metric,
                    "control": control,
                    "branch": branch,
                    "difference_branch_minus_control": branch - control,
                    "ratio_branch_over_control": (
                        branch / control if control != 0 else np.nan
                    ),
                }
            )
        if len(pair):
            differences = pair["branch"] - pair["control"]
            ratios = pair["branch"] / pair["control"].replace(0.0, np.nan)
            summary_rows.append(
                {
                    "metric": metric,
                    "n_pairs": len(pair),
                    "median_control": float(pair["control"].median()),
                    "median_branch": float(pair["branch"].median()),
                    "median_paired_difference": float(differences.median()),
                    "median_paired_ratio": float(
                        ratios.replace([np.inf, -np.inf], np.nan).median()
                    ),
                    "fraction_branch_greater": float((differences > 0).mean()),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


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

    cfg = EscapeDiagnosticConfig(control_window=20)
    events = build_first_passage_episode_table(
        daily,
        episodes,
        IntermediateProbeConfig(max_followup=cfg.max_followup),
    )
    events = events[events["resolved_within_followup"]].copy()

    # ------------------------------------------------------------------
    # 1. Residence time versus direction.
    # ------------------------------------------------------------------
    residence = residence_direction_summary(events)
    rank_effect = rank_biserial_recovery_minus_relapse(events)
    group_min_n = int(residence["n"].min()) if len(residence) else 0
    direction_material = bool(
        np.isfinite(rank_effect) and abs(rank_effect) >= DIRECTION_EFFECT_THRESHOLD
    )
    direction_verdict = verdict(direction_material, group_min_n)

    # ------------------------------------------------------------------
    # 2. Matched high-stress controls.
    # ------------------------------------------------------------------
    _, matches = match_high_stress_controls(daily, episodes, cfg)
    control_metrics = build_matched_control_relaxation(daily, events, matches, cfg)
    control_detail, control_summary = paired_control_summary(control_metrics)

    lag1_rows = control_summary[control_summary["metric"].str.endswith("__lag1")]
    lag1_positive = int(
        (lag1_rows["median_paired_difference"] >= CONTROL_LAG1_DELTA).sum()
    )
    shock_row = control_summary[
        control_summary["metric"] == "log_rv_ratio__shock_carryover_lag5"
    ]
    shock_ratio = (
        float(shock_row["median_paired_ratio"].iloc[0]) if len(shock_row) else np.nan
    )
    shock_support = bool(np.isfinite(shock_ratio) and shock_ratio >= CONTROL_SHOCK_RATIO)
    matched_support_count = lag1_positive + int(shock_support)
    matched_n = int(control_summary["n_pairs"].max()) if len(control_summary) else 0
    matched_verdict = verdict(
        matched_support_count >= CONTROL_REQUIRED_INDICATORS,
        matched_n,
    )

    # ------------------------------------------------------------------
    # 3. Resolution-aligned commitment window.
    # ------------------------------------------------------------------
    separation = build_resolution_aligned_separation(daily, events, cfg)
    commitment = commitment_window(separation, cfg)
    lead_summary = (
        separation.assign(
            abs_smd=lambda x: np.abs(x["smd_recovery_minus_relapse"]),
            strong=lambda x: np.abs(x["smd_recovery_minus_relapse"])
            >= cfg.commitment_smd_threshold,
        )
        .groupby("lead_days_before_escape", as_index=False)
        .agg(
            strong_feature_count=("strong", "sum"),
            median_abs_smd=("abs_smd", "median"),
            max_abs_smd=("abs_smd", "max"),
        )
        .sort_values("lead_days_before_escape", ascending=False)
    )

    verdicts = pd.DataFrame(
        [
            {
                "probe": "residence_time_depends_on_direction",
                "question": "Is |rank-biserial effect| for recovery minus relapse residence time >=0.33?",
                "key_value": rank_effect,
                "verdict": direction_verdict,
            },
            {
                "probe": "branch_relaxation_differs_from_matched_stress",
                "question": "Do at least 2 matched indicators support slower branch relaxation versus high-stress non-branch controls?",
                "key_value": matched_support_count,
                "verdict": matched_verdict,
            },
            {
                "probe": "resolution_aligned_commitment_window_detected",
                "question": "Is there a 3-day run where at least 2 of 4 path features have |SMD|>=0.8?",
                "key_value": commitment if commitment is not None else np.nan,
                "verdict": "YES" if commitment is not None else "NO",
            },
        ]
    )

    events.to_csv(args.output / "first_passage_episodes.csv", index=False)
    residence.to_csv(args.output / "residence_time_by_direction.csv", index=False)
    matches.to_csv(args.output / "matched_high_stress_controls.csv", index=False)
    control_metrics.to_csv(args.output / "matched_control_relaxation_metrics.csv", index=False)
    control_detail.to_csv(args.output / "matched_control_paired_detail.csv", index=False)
    control_summary.to_csv(args.output / "matched_control_summary.csv", index=False)
    separation.to_csv(args.output / "resolution_aligned_feature_separation.csv", index=False)
    lead_summary.to_csv(args.output / "resolution_aligned_lead_summary.csv", index=False)
    verdicts.to_csv(args.output / "probe_verdicts.csv", index=False)

    manifest = {
        "status": "final descriptive diagnostic block before model-design freeze",
        "probe_1": {
            "target": "first-passage residence time versus eventual direction",
            "effect": "rank-biserial recovery minus relapse",
            "screen": DIRECTION_EFFECT_THRESHOLD,
        },
        "probe_2": {
            "controls": "high-stress non-branch anchors, excluded within +/-120 rows of branch points, nearest-neighbor matched without replacement",
            "match_features": [
                "stress_ratio",
                "drawdown_120d",
                "rv_ratio_5_20_branch",
                "return_5d_branch",
            ],
            "window_days": cfg.control_window,
            "note": "descriptive full-sample matching; not a predictive/OOS evaluation",
        },
        "probe_3": {
            "alignment": "days backward from fixed-barrier first passage",
            "max_lead": cfg.commitment_max_lead,
            "features": list(COMMITMENT_FEATURES),
            "commitment_rule": (
                f">={cfg.commitment_required_metrics} features with |SMD|>="
                f"{cfg.commitment_smd_threshold} for {cfg.commitment_consecutive_days} consecutive days"
            ),
            "warning": "late separation is partly endpoint-conditioned; timing and sharpness are descriptive, not causal",
        },
        "verdicts_are_not_p_values": True,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Resolved first-passage episodes: {len(events)}")
    print("\nFinal three probes:")
    print(verdicts.to_string(index=False))
    print("\nResidence time by direction:")
    print(residence.to_string(index=False))
    print(f"Rank-biserial recovery-minus-relapse: {rank_effect:.6f}")
    print("\nMatched high-stress control summary:")
    print(control_summary.to_string(index=False))
    print("\nResolution-aligned lead summary:")
    print(lead_summary.to_string(index=False))
    print(f"\nDetected commitment window (farthest lead): {commitment}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
