from pathlib import Path
import pandas as pd

from qpitome_qrc.regimes.branch_intermediate_dynamics import IntermediateProbeConfig, build_first_passage_episode_table
from qpitome_qrc.regimes.branch_sync_clusters import cluster_branch_dates
from qpitome_qrc.regimes.crisis_clusters import CrisisClusterConfig, attach_event_dates, build_resolved_episode_intervals, cluster_episode_intervals, summarize_cluster_contribution
from qpitome_qrc.regimes.market_portability import DEFAULT_MARKETS, audit_market_portability

FOLLOWUP = 120
OUT = Path("results/diagnostics/cross_market_crisis_clusters_v2")
INPUTS = {
    "nikkei_225": Path("data/raw/portability/nikkei_225_fred_raw.csv"),
    "ftse_100": Path("data/raw/portability/ftse_100_raw.csv"),
    "russell_2000": Path("data/raw/portability/russell_2000_raw.csv"),
}


def complete(episodes, n_rows):
    return episodes[(n_rows - episodes["branch_idx"].astype(int) - 1) >= FOLLOWUP].copy()


def market_intervals(key, path):
    daily, episodes, events, _ = audit_market_portability(path, key)
    episodes = complete(episodes, len(daily))
    events = events[events["episode_id"].isin(episodes["episode_id"])].copy()
    events = attach_event_dates(daily, events)
    spec = DEFAULT_MARKETS[key]
    return build_resolved_episode_intervals(episodes, events, market_key=key, market=spec.market)


def spy_intervals():
    daily = pd.read_csv("data/processed/phase3_spy_vix_volatility_extended.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv("results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv", parse_dates=["branch_date"])
    episodes = complete(episodes, len(daily))
    events = build_first_passage_episode_table(daily, episodes, IntermediateProbeConfig(max_followup=FOLLOWUP))
    events = attach_event_dates(daily, events)
    return build_resolved_episode_intervals(episodes, events, market_key="spy", market="SPY")


def row(view, detail, clusters):
    return {
        "view": view,
        "n_resolved_market_episodes": len(detail),
        "n_calendar_clusters": len(clusters),
        "n_singleton_episode_clusters": int(clusters["singleton_episode"].sum()),
        "n_multi_market_clusters": int(clusters["multi_market"].sum()),
        "median_episodes_per_cluster": float(clusters["n_episodes"].median()),
        "max_episodes_in_cluster": int(clusters["n_episodes"].max()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [spy_intervals()] + [market_intervals(k, p) for k, p in INPUTS.items()]
    intervals = pd.concat(frames, ignore_index=True)
    sync_detail, sync_clusters = cluster_branch_dates(intervals, 30)
    broad_detail, broad_clusters = cluster_episode_intervals(intervals, CrisisClusterConfig(buffer_days=10))
    headlines = pd.DataFrame([
        row("anchored_branch_date_30d", sync_detail, sync_clusters),
        row("broad_interval_connected_10d_buffer", broad_detail, broad_clusters),
    ])
    sync_contribution = summarize_cluster_contribution(sync_detail)
    headlines.to_csv(OUT / "cluster_headline_comparison.csv", index=False)
    sync_detail.to_csv(OUT / "branch_sync_cluster_detail.csv", index=False)
    sync_clusters.to_csv(OUT / "branch_sync_clusters.csv", index=False)
    sync_contribution.to_csv(OUT / "branch_sync_market_contribution.csv", index=False)
    broad_clusters.to_csv(OUT / "broad_interval_clusters.csv", index=False)
    print("Cross-market crisis clustering v2")
    print("\nHeadline comparison:")
    print(headlines.to_string(index=False))
    print("\nPrimary anchored branch-date contribution:")
    print(sync_contribution.to_string(index=False))
    print("\nLargest 15 anchored branch-date clusters:")
    print(sync_clusters.sort_values("n_episodes", ascending=False).head(15).to_string(index=False))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
