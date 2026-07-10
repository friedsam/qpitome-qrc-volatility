import pandas as pd

from qpitome_qrc.regimes.branch_sync_clusters import cluster_branch_dates


def test_anchor_window_prevents_transitive_chaining():
    intervals = pd.DataFrame({
        "market_key": ["a", "b", "c"],
        "market": ["A", "B", "C"],
        "episode_id": [1, 1, 1],
        "branch_date": pd.to_datetime(["2024-01-01", "2024-01-25", "2024-02-15"]),
        "event_date": pd.to_datetime(["2024-01-10", "2024-02-01", "2024-02-20"]),
        "event_type": ["recovery", "relapse", "recovery"],
    })
    detail, summary = cluster_branch_dates(intervals, 30)
    assert detail["cluster_id"].tolist() == [1, 1, 2]
    assert summary["n_episodes"].tolist() == [2, 1]
