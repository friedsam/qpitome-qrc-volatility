from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.input_admission_assay import (
    InputAdmissionConfig,
    build_feature_tables,
    extract_market_window,
    parse_ticker,
    qlike_loss,
)


def test_parse_ticker_from_positive_and_negative_sample_ids() -> None:
    assert parse_ticker("P_GE001_^GSPC_data_L1") == "^GSPC"
    assert parse_ticker("N_P_GE054_000001.SS_data_L5_3") == "000001.SS"


def test_extract_market_window_builds_distinct_causal_channels() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D", tz="UTC")
    close = np.asarray([100.0, 99.0, 101.0, 98.0, 97.0, 100.0, 99.0, 102.0])
    panel = {
        "^TEST": pd.DataFrame(
            {
                "open": close * 0.995,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
            },
            index=dates,
        )
    }
    result = extract_market_window(
        panel,
        "^TEST",
        dates[-1],
        window=5,
        range_zero_fraction_limit=0.8,
    )
    assert result is not None
    assert result["downside_return"].shape == (5,)
    assert result["intraday_range"].shape == (5,)
    assert result["overnight_return"].shape == (5,)
    assert np.all(result["downside_return"] <= 0.0)
    assert bool(result["range_valid"][0])


def test_structurally_zero_range_is_flagged() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    close = np.linspace(100.0, 105.0, 6)
    panel = {
        "^TEST": pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
            },
            index=dates,
        )
    }
    result = extract_market_window(
        panel,
        "^TEST",
        dates[-1],
        window=5,
        range_zero_fraction_limit=0.8,
    )
    assert result is not None
    assert not bool(result["range_valid"][0])


def test_qlike_penalizes_underprediction_more_than_equal_overprediction() -> None:
    y = np.asarray([[1.0]])
    under = qlike_loss(y, np.asarray([[0.0]]))
    over = qlike_loss(y, np.asarray([[2.0]]))
    assert under > over


def test_feature_tables_preserve_rows_and_apply_range_gate() -> None:
    dates = pd.date_range("2020-01-01", periods=7, freq="D", tz="UTC")
    close = np.linspace(100.0, 106.0, 7)
    panel = {
        "^TEST": pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
            },
            index=dates,
        )
    }
    manifest = pd.DataFrame(
        {
            "fold": [4],
            "sample_id": ["P_GE001_^TEST_data_L1"],
            "lead": [1],
            "label": [1],
            "ticker": ["^TEST"],
            "origin_date": [dates[-1]],
        }
    )
    config = InputAdmissionConfig(window=5)
    matrices, availability, names = build_feature_tables(manifest, panel, config)
    assert matrices[("all_three", "sequence")].shape == (1, 15)
    assert matrices[("all_three", "summary")].shape[0] == 1
    assert availability["valid__all_three"].iat[0]
    assert len(names[("downside_return", "sequence")]) == 5
