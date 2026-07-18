from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/run_cross_market_window_availability.py")
SPEC = importlib.util.spec_from_file_location("run_cross_market_window_availability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_count_available_markets_requires_history_and_freshness() -> None:
    dates = {
        "A": pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC").to_numpy(dtype="datetime64[ns]"),
        "B": pd.date_range("2020-02-01", periods=10, freq="D", tz="UTC").to_numpy(dtype="datetime64[ns]"),
        "C": pd.date_range("2019-01-01", periods=50, freq="D", tz="UTC").to_numpy(dtype="datetime64[ns]"),
    }
    count, tickers = MODULE.count_available_markets(
        pd.Timestamp("2020-02-19", tz="UTC"),
        dates,
        required_observations=40,
        max_staleness_days=7,
    )
    assert count == 1
    assert tickers == ["A"]


def test_audit_windows_reports_target_mapping() -> None:
    panel_dates = {
        "A": pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC").to_numpy(dtype="datetime64[ns]"),
    }
    manifest = pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "origin_date": ["2020-02-19", "2020-02-19"],
        "market_group": ["A", "UNKNOWN"],
    })
    audit = MODULE.audit_windows(manifest, panel_dates, required_observations=40, max_staleness_days=7)
    assert audit["available_markets"].tolist() == [1, 1]
    assert audit["target_exact_ticker_match"].tolist() == [True, False]
    assert audit["target_available"].tolist() == [True, False]
