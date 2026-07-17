from __future__ import annotations

from pathlib import Path

import pandas as pd

from transition_forecasting.catalogue.global_transition_validation import validate_global_catalogue


def test_validate_global_catalogue_reports_cluster_and_exposure_metrics(tmp_path: Path) -> None:
    inventory = pd.DataFrame(
        [
            {"index": "A", "eligible": True, "start": "2000-01-01", "end": "2019-12-31"},
            {"index": "B", "eligible": True, "start": "2010-01-01", "end": "2019-12-31"},
        ]
    )
    raw = pd.DataFrame(
        [
            {"index": "A", "onset_date": "2011-01-01"},
            {"index": "B", "onset_date": "2011-01-04"},
            {"index": "A", "onset_date": "2015-01-01"},
        ]
    )
    representative = pd.DataFrame(
        [
            {"index": "A", "market_group": "m1", "episode_id": "GE001", "onset_date": "2011-01-01"},
            {"index": "B", "market_group": "m2", "episode_id": "GE001", "onset_date": "2011-01-04"},
            {"index": "A", "market_group": "m1", "episode_id": "GE002", "onset_date": "2015-01-01"},
        ]
    )
    episodes = pd.DataFrame(
        [
            {"episode_id": "GE001", "start_date": "2011-01-01", "end_date": "2011-01-04", "n_markets": 2, "n_index_events": 2},
            {"episode_id": "GE002", "start_date": "2015-01-01", "end_date": "2015-01-01", "n_markets": 1, "n_index_events": 1},
        ]
    )

    paths = []
    for name, frame in [("inventory.csv", inventory), ("raw.csv", raw), ("representative.csv", representative), ("episodes.csv", episodes)]:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths.append(path)

    summary, distribution, decade_rates = validate_global_catalogue(*paths)

    assert summary["global_episodes"] == 2
    assert summary["singleton_market_episodes"] == 1
    assert summary["multi_market_episodes"] == 1
    assert summary["maximum_cluster_span_days"] == 3
    assert int(distribution["episodes"].sum()) == 2
    assert decade_rates.loc[decade_rates["decade"] == 2010, "events"].iloc[0] == 3
    assert decade_rates["events_per_100_market_years"].notna().all()
