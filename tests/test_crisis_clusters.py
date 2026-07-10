from __future__ import annotations

import pandas as pd

from qpitome_qrc.regimes.crisis_clusters import (
    CrisisClusterConfig,
    attach_event_dates,
    cluster_episode_intervals,
    summarize_cluster_contribution,
)


def test_attach_event_dates_uses_exact_trading_row() -> None:
    daily = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])}
    )
    events = pd.DataFrame(
        {
            "branch_idx": [0, 1],
            "event_day": [2, 1],
            "resolved_within_followup": [True, False],
        }
    )

    out = attach_event_dates(daily, events)

    assert out.loc[0, "event_date"] == pd.Timestamp("2024-01-05")
    assert pd.isna(out.loc[1, "event_date"])


def test_cluster_episode_intervals_forms_connected_components() -> None:
    intervals = pd.DataFrame(
        {
            "market_key": ["a", "b", "c", "d"],
            "market": ["A", "B", "C", "D"],
            "episode_id": [1, 1, 1, 1],
            "branch_date": pd.to_datetime(
                ["2024-01-01", "2024-01-12", "2024-01-25", "2024-04-01"]
            ),
            "event_date": pd.to_datetime(
                ["2024-01-10", "2024-01-20", "2024-02-01", "2024-04-10"]
            ),
            "event_type": ["recovery", "relapse", "recovery", "relapse"],
            "event_day": [5, 4, 6, 5],
        }
    )

    detail, summary = cluster_episode_intervals(
        intervals, CrisisClusterConfig(buffer_days=2)
    )

    assert detail["cluster_id"].tolist() == [1, 1, 1, 2]
    assert summary["n_episodes"].tolist() == [3, 1]
    assert summary["n_markets"].tolist() == [3, 1]
    assert summary["multi_market"].tolist() == [True, False]


def test_summarize_cluster_contribution_counts_market_unique_clusters() -> None:
    detail = pd.DataFrame(
        {
            "market_key": ["a", "a", "b", "b"],
            "market": ["A", "A", "B", "B"],
            "episode_id": [1, 2, 1, 2],
            "cluster_id": [1, 2, 2, 3],
        }
    )

    summary = summarize_cluster_contribution(detail).set_index("market_key")

    assert summary.loc["a", "n_calendar_clusters_touched"] == 2
    assert summary.loc["a", "n_market_unique_clusters"] == 1
    assert summary.loc["b", "n_calendar_clusters_touched"] == 2
    assert summary.loc["b", "n_market_unique_clusters"] == 1
