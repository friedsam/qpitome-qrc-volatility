"""Audit calendar dependence among SPY and cross-market branch episodes.

The detector-portability audit counts market episodes. This audit asks a different
question: how many distinct calendar crisis clusters do those episodes represent?
Resolved branch-to-first-passage intervals are padded by a fixed calendar buffer and
grouped by connected overlap. The output is a dependence sensitivity, not a claim
that episodes within different clusters are statistically independent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    IntermediateProbeConfig,
    build_first_passage_episode_table,
)
from qpitome_qrc.regimes.crisis_clusters import (
    CrisisClusterConfig,
    attach_event_dates,
    build_resolved_episode_intervals,
    cluster_episode_intervals,
    summarize_cluster_contribution,
)
from qpitome_qrc.regimes.market_portability import (
    DEFAULT_MARKETS,
    audit_market_portability,
)


DEFAULT_INPUTS = {
    "nikkei_225": Path("data/raw/portability/nikkei_225_fred_raw.csv"),
    "ftse_100": Path("data/raw/portability/ftse_100_raw.csv"),
    "russell_2000": Path("data/raw/portability/russell_2000_raw.csv"),
}
DEFAULT_SPY_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_SPY_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path("results/diagnostics/cross_market_crisis_clusters_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nikkei", type=Path, default=DEFAULT_INPUTS["nikkei_225"])
    parser.add_argument("--ftse", type=Path, default=DEFAULT_INPUTS["ftse_100"])
    parser.add_argument("--russell", type=Path, default=DEFAULT_INPUTS["russell_2000"])
    parser.add_argument("--spy-data", type=Path, default=DEFAULT_SPY_DATA)
    parser.add_argument("--spy-episodes", type=Path, default=DEFAULT_SPY_EPISODES)
    parser.add_argument("--buffer-days", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _market_intervals(market_key: str, path: Path) -> pd.DataFrame:
    daily, episodes, events, _ = audit_market_portability(path, market_key)
    events = attach_event_dates(daily, events)
    spec = DEFAULT_MARKETS[market_key]
    return build_resolved_episode_intervals(
        episodes,
        events,
        market_key=market_key,
        market=spec.market,
    )


def _spy_intervals(data_path: Path, episode_path: Path) -> pd.DataFrame:
    daily = (
        pd.read_csv(data_path, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    episodes = pd.read_csv(episode_path, parse_dates=["branch_date"])
    if "outcome_complete" in episodes.columns:
        episodes = episodes[episodes["outcome_complete"].astype(bool)].copy()
    events = build_first_passage_episode_table(
        daily,
        episodes,
        IntermediateProbeConfig(max_followup=120),
    )
    events = attach_event_dates(daily, events)
    return build_resolved_episode_intervals(
        episodes,
        events,
        market_key="spy",
        market="SPY",
    )


def main() -> None:
    args = parse_args()
    inputs = {
        "nikkei_225": args.nikkei,
        "ftse_100": args.ftse,
        "russell_2000": args.russell,
    }
    for path in [*inputs.values(), args.spy_data, args.spy_episodes]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.buffer_days < 0:
        raise ValueError("buffer-days must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)

    interval_frames = [_spy_intervals(args.spy_data, args.spy_episodes)]
    interval_frames.extend(
        _market_intervals(market_key, path) for market_key, path in inputs.items()
    )
    intervals = pd.concat(interval_frames, ignore_index=True)

    cfg = CrisisClusterConfig(buffer_days=args.buffer_days)
    detail, clusters = cluster_episode_intervals(intervals, cfg)
    contribution = summarize_cluster_contribution(detail)

    annual = (
        detail.assign(branch_year=lambda frame: frame["branch_date"].dt.year)
        .groupby(["branch_year", "market_key"], as_index=False)
        .size()
        .rename(columns={"size": "n_episodes"})
        .sort_values(["branch_year", "market_key"])
    )
    decade = (
        detail.assign(decade=lambda frame: (frame["branch_date"].dt.year // 10) * 10)
        .groupby(["decade", "market_key"], as_index=False)
        .size()
        .rename(columns={"size": "n_episodes"})
        .sort_values(["decade", "market_key"])
    )

    headline = pd.DataFrame(
        [
            {
                "n_resolved_market_episodes": len(detail),
                "n_calendar_crisis_clusters": len(clusters),
                "n_singleton_episode_clusters": int(clusters["singleton_episode"].sum()),
                "n_multi_market_clusters": int(clusters["multi_market"].sum()),
                "fraction_clusters_multi_market": float(clusters["multi_market"].mean()),
                "median_episodes_per_cluster": float(clusters["n_episodes"].median()),
                "max_episodes_in_cluster": int(clusters["n_episodes"].max()),
            }
        ]
    )

    detail.to_csv(args.output / "clustered_episode_detail.csv", index=False)
    clusters.to_csv(args.output / "calendar_crisis_clusters.csv", index=False)
    contribution.to_csv(args.output / "market_cluster_contribution.csv", index=False)
    annual.to_csv(args.output / "episode_counts_by_year.csv", index=False)
    decade.to_csv(args.output / "episode_counts_by_decade.csv", index=False)
    headline.to_csv(args.output / "cluster_headline_summary.csv", index=False)

    manifest = {
        "status": "cross-market calendar-dependence sensitivity",
        "markets": ["spy", "nikkei_225", "ftse_100", "russell_2000"],
        "interval": "branch date through exact fixed-barrier first-passage date",
        "buffer_calendar_days": args.buffer_days,
        "cluster_rule": "connected components of overlapping buffered episode intervals",
        "warning": "Calendar clusters are a dependence sensitivity. Distinct clusters are not asserted statistically independent, and same-cluster episodes remain valid market-level observations.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Cross-market calendar crisis clustering")
    print(headline.to_string(index=False))
    print("\nMarket contribution:")
    print(contribution.to_string(index=False))
    print("\nLargest 15 clusters:")
    print(
        clusters.sort_values(["n_episodes", "n_markets"], ascending=False)
        .head(15)
        .to_string(index=False)
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
