import pytest

from data.download_market_data import fetch_yahoo_history


def test_fetch_rejects_nonincreasing_date_range() -> None:
    with pytest.raises(ValueError, match="end_date"):
        fetch_yahoo_history("SPY", "2024-01-02", "2024-01-02")
