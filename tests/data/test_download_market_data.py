from pathlib import Path

import pandas as pd
import pytest

from data.download_market_data import download_market_data, fetch_yahoo_history


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1_000_000, 1_100_000],
            "adjusted_close": [100.4, 101.4],
        }
    )


def _volatility_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [15.0, 16.0],
            "high": [16.0, 17.0],
            "low": [14.0, 15.0],
            "close": [15.5, 16.5],
            "volume": [0, 0],
        }
    )


def test_fetch_rejects_nonincreasing_date_range() -> None:
    with pytest.raises(ValueError, match="end_date"):
        fetch_yahoo_history("SPY", "2024-01-02", "2024-01-02")


def test_download_market_data_copies_explicit_fallbacks(tmp_path: Path) -> None:
    market_fallback = tmp_path / "fallback" / "market.csv"
    volatility_fallback = tmp_path / "fallback" / "volatility.csv"
    market_fallback.parent.mkdir(parents=True)
    _market_frame().to_csv(market_fallback, index=False)
    _volatility_frame().to_csv(volatility_fallback, index=False)

    def failing_fetcher(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise RuntimeError(f"offline: {symbol}")

    market_output = tmp_path / "raw" / "market.csv"
    volatility_output = tmp_path / "raw" / "volatility.csv"
    result = download_market_data(
        market_symbol="SPY",
        volatility_symbol="^VIX",
        start_date="2024-01-01",
        end_date="2024-02-01",
        market_output_path=market_output,
        volatility_output_path=volatility_output,
        market_fallback_path=market_fallback,
        volatility_fallback_path=volatility_fallback,
        manifest_path=tmp_path / "raw" / "manifest.json",
        history_fetcher=failing_fetcher,
    )

    assert result.market_status == "fallback_copied"
    assert result.volatility_status == "fallback_copied"
    assert market_output.read_bytes() == market_fallback.read_bytes()
    assert volatility_output.read_bytes() == volatility_fallback.read_bytes()


def test_download_market_data_rejects_invalid_market_fallback(
    tmp_path: Path,
) -> None:
    market_fallback = tmp_path / "fallback" / "market.csv"
    volatility_fallback = tmp_path / "fallback" / "volatility.csv"
    market_fallback.parent.mkdir(parents=True)
    _market_frame().drop(columns=["adjusted_close"]).to_csv(
        market_fallback,
        index=False,
    )
    _volatility_frame().to_csv(volatility_fallback, index=False)

    def failing_fetcher(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise RuntimeError("offline")

    with pytest.raises(RuntimeError, match="fallback is unavailable or invalid"):
        download_market_data(
            market_symbol="SPY",
            volatility_symbol="^VIX",
            start_date="2024-01-01",
            end_date="2024-02-01",
            market_output_path=tmp_path / "raw" / "market.csv",
            volatility_output_path=tmp_path / "raw" / "volatility.csv",
            market_fallback_path=market_fallback,
            volatility_fallback_path=volatility_fallback,
            manifest_path=tmp_path / "raw" / "manifest.json",
            history_fetcher=failing_fetcher,
        )
