"""Calendar clustering for synchronized cross-market branch episodes.

Market episodes are distinct detector observations but need not be independent crisis
realizations. This module groups overlapping branch-to-first-passage intervals into
calendar-connected components without altering the detector or outcome definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CrisisClusterConfig:
    """Calendar-overlap rule for cross-market dependence sensitivity."""

    buffer_days: int = 10


def build_resolved_episode_intervals(
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    *,
    market_key: str,
    market: str,
) -> pd.DataFrame:
    """Return resolved branch-to-first-passage calendar intervals for one market."""
    required_episode = {"episode_id", "branch_date"}
    missing = required_episode - set(episodes.columns)
    if missing:
        raise KeyError(f"Missing episode columns: {sorted(missing)}")
    required_event = {
        "episode_id",
        "event_day",
        "event_type",
        "resolved_within_followup",
    }
    missing = required_event - set(events.columns)
    if missing:
        raise KeyError(f"Missing event columns: {sorted(missing)}")

    merged = episodes[["episode_id", "branch_date"]].merge(
        events[list(required_event)], on="episode_id", how="inner", validate="one_to_one"
    )
    merged = merged[merged["resolved_within_followup"].astype(bool)].copy()
    merged["branch_date"] = pd.to_datetime(merged["branch_date"])
    merged["event_day"] = pd.to_numeric(merged["event_day"], errors="coerce")
    merged = merged.dropna(subset=["branch_date", "event_day", "event_type"])

    # Event day is in market trading rows, so the actual event date must be supplied
    # by the caller before clustering when available. A calendar approximation would
    # distort cross-market overlap. This function therefore requires event_date in
    # events when converting intervals.
    if "event_date" not in events.columns:
        raise KeyError("event_date")
    event_dates = events[["episode_id", "event_date"]].copy()
    event_dates["event_date"] = pd.to_datetime(event_dates["event_date"])
    merged = merged.merge(event_dates, on="episode_id", how="left", validate="one_to_one")
    merged = merged.dropna(subset=["event_date"])

    merged["market_key"] = market_key
    merged["market"] = market
    return merged[
        [
            "market_key",
            "market",
            "episode_id",
            "branch_date",
            "event_date",
            "event_day",
            "event_type",
        ]
    ].sort_values(["branch_date", "event_date"]).reset_index(drop=True)


def attach_event_dates(
    daily: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact trading-calendar first-passage dates to an event table."""
    required_daily = {"date"}
    missing = required_daily - set(daily.columns)
    if missing:
        raise KeyError(f"Missing daily columns: {sorted(missing)}")
    required_event = {"branch_idx", "event_day", "resolved_within_followup"}
    missing = required_event - set(events.columns)
    if missing:
        raise KeyError(f"Missing event columns: {sorted(missing)}")

    dates = pd.to_datetime(daily["date"]).reset_index(drop=True)
    out = events.copy()
    event_dates: list[pd.Timestamp | pd.NaT] = []
    for row in out.itertuples(index=False):
        if not bool(row.resolved_within_followup) or pd.isna(row.event_day):
            event_dates.append(pd.NaT)
            continue
        event_idx = int(row.branch_idx) + int(row.event_day)
        event_dates.append(dates.iloc[event_idx] if 0 <= event_idx < len(dates) else pd.NaT)
    out["event_date"] = event_dates
    return out


def cluster_episode_intervals(
    intervals: pd.DataFrame,
    config: CrisisClusterConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group buffered overlapping intervals into calendar-connected components."""
    cfg = config or CrisisClusterConfig()
    required = {
        "market_key",
        "market",
        "episode_id",
        "branch_date",
        "event_date",
        "event_type",
    }
    missing = required - set(intervals.columns)
    if missing:
        raise KeyError(f"Missing interval columns: {sorted(missing)}")
    if intervals.empty:
        return intervals.assign(cluster_id=pd.Series(dtype=int)), pd.DataFrame()

    work = intervals.copy()
    work["branch_date"] = pd.to_datetime(work["branch_date"])
    work["event_date"] = pd.to_datetime(work["event_date"])
    delta = pd.to_timedelta(cfg.buffer_days, unit="D")
    work["cluster_start_candidate"] = work["branch_date"] - delta
    work["cluster_end_candidate"] = work["event_date"] + delta
    work = work.sort_values(
        ["cluster_start_candidate", "cluster_end_candidate", "market_key", "episode_id"]
    ).reset_index(drop=True)

    cluster_ids: list[int] = []
    cluster_id = 0
    current_end: pd.Timestamp | None = None
    for row in work.itertuples(index=False):
        start = row.cluster_start_candidate
        end = row.cluster_end_candidate
        if current_end is None or start > current_end:
            cluster_id += 1
            current_end = end
        else:
            current_end = max(current_end, end)
        cluster_ids.append(cluster_id)
    work["cluster_id"] = cluster_ids

    summary = (
        work.groupby("cluster_id", as_index=False)
        .agg(
            cluster_start=("branch_date", "min"),
            cluster_end=("event_date", "max"),
            n_episodes=("episode_id", "size"),
            n_markets=("market_key", "nunique"),
            markets=("market_key", lambda values: "|".join(sorted(set(values)))),
            n_recovery=("event_type", lambda values: int((values == "recovery").sum())),
            n_relapse=("event_type", lambda values: int((values == "relapse").sum())),
        )
        .sort_values("cluster_start")
        .reset_index(drop=True)
    )
    summary["multi_market"] = summary["n_markets"] > 1
    summary["singleton_episode"] = summary["n_episodes"] == 1

    detail = work.drop(columns=["cluster_start_candidate", "cluster_end_candidate"])
    return detail, summary


def summarize_cluster_contribution(detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize total versus calendar-unique cluster contribution by market."""
    required = {"market_key", "market", "episode_id", "cluster_id"}
    missing = required - set(detail.columns)
    if missing:
        raise KeyError(f"Missing clustered-detail columns: {sorted(missing)}")

    cluster_sizes = detail.groupby("cluster_id")["market_key"].nunique()
    rows: list[dict[str, object]] = []
    for (market_key, market), group in detail.groupby(["market_key", "market"]):
        cluster_ids = set(group["cluster_id"].astype(int))
        unique_clusters = sum(cluster_sizes.loc[cluster_id] == 1 for cluster_id in cluster_ids)
        rows.append(
            {
                "market_key": market_key,
                "market": market,
                "n_resolved_episodes": len(group),
                "n_calendar_clusters_touched": len(cluster_ids),
                "n_market_unique_clusters": int(unique_clusters),
                "fraction_clusters_market_unique": (
                    float(unique_clusters / len(cluster_ids)) if cluster_ids else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("market_key").reset_index(drop=True)
