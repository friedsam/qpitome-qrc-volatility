"""Audit the formal branching-state extractor before locking a benchmark.

The exploratory work established the detector morphology but did not preserve
its exact numeric threshold grid. This runner therefore performs a transparent
24-definition robustness audit around the mid-grid candidate and writes every
candidate episode set for inspection.

No model fitting occurs here.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from qpitome_qrc.regimes.branching_state import (
    BranchStateConfig,
    OutcomeConfig,
    config_manifest,
    extract_labeled_branch_episodes,
)


DEFAULT_EXTENDED_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_FROZEN_DATA = Path("data/processed/phase2_spy_vix_volatility.csv")
DEFAULT_OUTPUT = Path("results/regimes/branching_extractor_audit_v1")

# 3 x 2 x 2 x 2 = 24 definitions. The dimensions correspond to the four
# threshold choices that most directly control the state morphology described
# in regime_branching_analysis.md.
STRESS_QUANTILES = [0.65, 0.70, 0.75]
DRAWDOWN_THRESHOLDS = [-0.05, -0.06]
STABILIZATION_FLOORS = [-0.015, -0.005]
PRIOR_DECLINE_THRESHOLDS = [-0.03, -0.05]

MID_GRID = BranchStateConfig(
    stress_quantile=0.70,
    drawdown_threshold=-0.06,
    rv_ratio_cap=1.00,
    stabilization_return_floor=-0.005,
    prior_decline_threshold=-0.05,
    prior_decline_lookback=20,
    merge_gap_days=3,
    min_episode_separation=20,
)

OUTCOME_CONFIG = OutcomeConfig(
    horizon=40,
    recovery_scale=0.55,
    relapse_scale=0.70,
)


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
    raise FileNotFoundError(
        f"Neither {DEFAULT_EXTENDED_DATA} nor {DEFAULT_FROZEN_DATA} exists"
    )


def config_id(cfg: BranchStateConfig) -> str:
    return (
        f"q{int(round(cfg.stress_quantile * 100)):02d}"
        f"_dd{int(round(abs(cfg.drawdown_threshold) * 100)):02d}"
        f"_stab{int(round((cfg.stabilization_return_floor + 0.02) * 1000)):02d}"
        f"_decl{int(round(abs(cfg.prior_decline_threshold) * 100)):02d}"
    )


def summarize_config(
    cfg_id: str,
    cfg: BranchStateConfig,
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, object]:
    labeled = episodes[episodes["outcome_complete"]].copy()
    counts = labeled["outcome"].value_counts()
    years = pd.to_datetime(labeled["branch_date"]).dt.year if not labeled.empty else pd.Series(dtype=int)

    return {
        "config_id": cfg_id,
        "stress_quantile": cfg.stress_quantile,
        "drawdown_threshold": cfg.drawdown_threshold,
        "stabilization_return_floor": cfg.stabilization_return_floor,
        "prior_decline_threshold": cfg.prior_decline_threshold,
        "candidate_days": int(daily["branch_candidate"].sum()),
        "episodes_total": int(len(episodes)),
        "episodes_labeled": int(len(labeled)),
        "recovery": int(counts.get("recovery", 0)),
        "relapse": int(counts.get("relapse", 0)),
        "mixed": int(counts.get("mixed", 0)),
        "first_branch_year": int(years.min()) if len(years) else None,
        "last_branch_year": int(years.max()) if len(years) else None,
        "distinct_5y_blocks": int((years // 5).nunique()) if len(years) else 0,
    }


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output = args.output
    episode_dir = output / "episode_sets"
    output.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["date"])
    summaries: list[dict[str, object]] = []

    for stress_q, dd, stab, decline in itertools.product(
        STRESS_QUANTILES,
        DRAWDOWN_THRESHOLDS,
        STABILIZATION_FLOORS,
        PRIOR_DECLINE_THRESHOLDS,
    ):
        cfg = BranchStateConfig(
            stress_quantile=stress_q,
            drawdown_threshold=dd,
            rv_ratio_cap=1.00,
            stabilization_return_floor=stab,
            prior_decline_threshold=decline,
            prior_decline_lookback=20,
            merge_gap_days=3,
            min_episode_separation=20,
        )
        cid = config_id(cfg)
        daily, episodes = extract_labeled_branch_episodes(df, cfg, OUTCOME_CONFIG)
        episodes.to_csv(episode_dir / f"{cid}.csv", index=False)
        summaries.append(summarize_config(cid, cfg, daily, episodes))

    summary = pd.DataFrame(summaries).sort_values(
        ["episodes_labeled", "mixed", "config_id"],
        ascending=[False, True, True],
    )
    summary.to_csv(output / "threshold_audit_summary.csv", index=False)

    mid_id = config_id(MID_GRID)
    mid_daily, mid_episodes = extract_labeled_branch_episodes(
        df, MID_GRID, OUTCOME_CONFIG
    )
    mid_daily.to_csv(output / "mid_grid_daily_candidates.csv", index=False)
    mid_episodes.to_csv(output / "mid_grid_episodes.csv", index=False)

    decade_counts = (
        mid_episodes[mid_episodes["outcome_complete"]]
        .assign(decade=lambda x: (pd.to_datetime(x["branch_date"]).dt.year // 10) * 10)
        .groupby(["decade", "outcome"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    decade_counts.to_csv(output / "mid_grid_decade_outcomes.csv", index=False)

    manifest = {
        "data_path": str(data_path),
        "n_rows": len(df),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "grid": {
            "stress_quantiles": STRESS_QUANTILES,
            "drawdown_thresholds": DRAWDOWN_THRESHOLDS,
            "stabilization_return_floors": STABILIZATION_FLOORS,
            "prior_decline_thresholds": PRIOR_DECLINE_THRESHOLDS,
            "n_definitions": 24,
        },
        "mid_grid": config_manifest(MID_GRID, OUTCOME_CONFIG),
        "status": "audit only; no benchmark definition is locked by this runner",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Data: {data_path}")
    print(f"Rows: {len(df)}")
    print(f"Range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print("\n24-definition robustness summary:")
    print(summary.to_string(index=False))
    print(f"\nMid-grid config: {mid_id}")
    print(mid_episodes[[
        "episode_id",
        "branch_date",
        "outcome",
        "rv_20d",
        "drawdown_120d",
        "rv_ratio_5_20_branch",
        "return_5d_branch",
        "forward_return_h",
        "worst_forward_drawdown_h",
    ]].to_string(index=False))
    print("\nMid-grid decade outcomes:")
    print(decade_counts.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
