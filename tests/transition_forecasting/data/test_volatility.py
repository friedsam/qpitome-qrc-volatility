from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.data.volatility import (
    build_daily_volatility,
    consolidate_cleaned_ohlc,
)


def test_consolidate_cleaned_ohlc_orders_indices_and_dates(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-02"],
            "open": [101.0, 100.0],
            "high": [102.0, 101.0],
            "low": [100.0, 99.0],
            "close": [101.5, 100.5],
        }
    ).to_csv(cleaned / "B.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "open": [200.0],
            "high": [202.0],
            "low": [198.0],
            "close": [201.0],
        }
    ).to_csv(cleaned / "A.csv", index=False)

    combined = consolidate_cleaned_ohlc(cleaned, tmp_path / "cleaned_ohlc.csv.gz")

    assert list(zip(combined["index"], combined["date"].dt.strftime("%Y-%m-%d"))) == [
        ("A", "2020-01-02"),
        ("B", "2020-01-02"),
        ("B", "2020-01-03"),
    ]


def test_build_daily_volatility_applies_effective_start(tmp_path: Path) -> None:
    source = tmp_path / "INDEX.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 102.0, 104.0],
            "low": [99.0, 98.0, 96.0],
            "close": [100.0, 100.0, 100.0],
        }
    ).to_csv(source, index=False)
    quality = tmp_path / "range_quality.csv"
    pd.DataFrame(
        {
            "index": ["INDEX"],
            "path": [str(source)],
            "recommended_effective_start": ["2020-01-02"],
        }
    ).to_csv(quality, index=False)

    result = build_daily_volatility(quality, tmp_path / "daily_volatility.csv.gz")

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-02",
        "2020-01-03",
    ]
    expected = np.log(
        np.abs(np.log(np.array([102.0 / 98.0, 104.0 / 96.0])))
        / np.sqrt(4 * np.log(2))
    )
    np.testing.assert_allclose(result["log_parkinson_volatility"], expected)
