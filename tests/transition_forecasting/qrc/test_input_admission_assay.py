from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.input_admission_assay import (
    InputAdmissionConfig,
    build_feature_tables,
    extract_market_window,
    fit_select_signed_residual,
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


def test_signed_residual_fit_ignores_nan_rows_outside_eligible_masks() -> None:
    rng = np.random.default_rng(20260724)
    rows = 24
    features = rng.normal(size=(rows, 4))
    features[18:] = np.nan
    residuals = rng.normal(scale=0.1, size=(rows, 10))
    har = rng.normal(size=(rows, 10))
    targets = har + residuals
    dates = pd.Series(pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC"))
    fit_eligible = np.zeros(rows, dtype=bool)
    fit_eligible[:16] = True
    prediction_eligible = np.zeros(rows, dtype=bool)
    prediction_eligible[:18] = True

    prediction, selected, candidates = fit_select_signed_residual(
        features,
        residuals,
        har,
        targets,
        dates,
        fit_eligible,
        prediction_eligible,
        InputAdmissionConfig(alphas=(0.1, 1.0), inner_holdout_fraction=0.25),
    )

    assert np.isfinite(prediction[:18]).all()
    assert np.isnan(prediction[18:]).all()
    assert selected["alpha"] in {0.1, 1.0}
    assert len(candidates) == 2
