from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.ohlc_range_quality import annual_range_quality


def test_annual_range_quality_recommends_first_qualifying_year(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "date": ["1949-01-03", "1949-01-04", "1950-01-03", "1950-01-04"],
            "high": [10.0, 10.0, 11.0, 12.0],
            "low": [10.0, 10.0, 10.0, 11.0],
        }
    ).to_csv(path, index=False)

    annual, summary = annual_range_quality(path, minimum_nonzero_fraction=0.95)

    assert summary["first_nonzero_range_date"] == "1950-01-03"
    assert summary["first_qualifying_year"] == 1950
    assert summary["recommended_effective_start"] == "1950-01-01"
    assert summary["longest_consecutive_zero_range_run"] == 2
    assert annual.loc[annual["year"] == 1949, "meets_quality_threshold"].iloc[0] == False
    assert annual.loc[annual["year"] == 1950, "meets_quality_threshold"].iloc[0] == True
