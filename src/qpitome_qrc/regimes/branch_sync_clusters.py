from __future__ import annotations

import pandas as pd


def cluster_branch_dates(intervals: pd.DataFrame, sync_days: int = 30):
    if sync_days < 0:
        raise ValueError("sync_days must be nonnegative")
    work = intervals.copy()
    work["branch_date"] = pd.to_datetime(work["branch_date"])
    work = work.sort_values(["branch_date", "market_key", "episode_id"]).reset_index(drop=True)
    window = pd.to_timedelta(sync_days, unit="D")
    cluster_ids = []
    cluster_id = 0
    anchor = None
    for row in work.itertuples(index=False):
        if anchor is None or row.branch_date > anchor + window:
            cluster_id += 1
            anchor = row.branch_date
        cluster_ids.append(cluster_id)
    work["cluster_id"] = cluster_ids
    summary = work.groupby("cluster_id", as_index=False).agg(
        cluster_start=("branch_date", "min"),
        cluster_end=("branch_date", "max"),
        n_episodes=("episode_id", "size"),
        n_markets=("market_key", "nunique"),
    )
    summary["multi_market"] = summary["n_markets"] > 1
    summary["singleton_episode"] = summary["n_episodes"] == 1
    return work, summary
