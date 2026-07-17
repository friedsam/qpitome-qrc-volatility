from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.early_ohlc_quality import audit_early_ohlc_file
from data.global_transition_catalogue import _cluster_dates


def test_cluster_dates_uses_fixed_window_anchor() -> None:
    catalogue = pd.DataFrame(
        {"onset_date": pd.to_datetime(["2020-01-01", "2020-01-07", "2020-01-13"])}
    )
    clustered = _cluster_dates(catalogue)
    assert clustered["episode_id"].tolist() == ["GE001", "GE001", "GE002"]


def test_early_ohlc_audit_detects_repeated_values(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "date": ["1927-12-30", "1928-01-03", "1928-01-04", "1951-01-01"],
            "open": [10.0, 10.0, 11.0, 12.0],
            "high": [10.5, 10.5, 11.5, 12.5],
            "low": [9.5, 9.5, 10.5, 11.5],
            "close": [10.0, 10.0, 11.0, 12.0],
        }
    ).to_csv(path, index=False)

    summary, runs = audit_early_ohlc_file(path)

    assert summary["early_rows"] == 3
    assert summary["consecutive_identical_value_rows"] == 1
    assert summary["maximum_identical_run_length"] == 2
    assert len(runs) == 1
