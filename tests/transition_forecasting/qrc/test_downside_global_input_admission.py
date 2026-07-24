from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.downside_global_input_admission import (
    DownsideGlobalAdmissionConfig,
    build_leave_one_out_global_breadth,
    build_local_state,
    fit_select_occurrence,
)


def _market_frame(returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(np.asarray(returns, dtype=float)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=dates,
    )


def test_global_breadth_is_strictly_lagged_and_excludes_target_market() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    alternating = np.asarray([0.01 if index % 2 == 0 else -0.005 for index in range(40)])
    positive = np.asarray([0.008 if index % 2 == 0 else 0.004 for index in range(40)])
    a = alternating.copy()
    b = positive.copy()
    c = positive.copy()
    shock_index = 25
    a[shock_index] = -0.15
    panel = {
        "A": _market_frame(a, dates),
        "B": _market_frame(b, dates),
        "C": _market_frame(c, dates),
        "^VIX": _market_frame(np.full(40, -0.20), dates),
    }
    config = DownsideGlobalAdmissionConfig(
        window=10,
        drawdown_lookback=10,
        local_volatility_lookback=5,
        minimum_peer_markets=2,
    )
    breadth = build_leave_one_out_global_breadth(panel, config)

    # A's crash is absent from A's own leave-one-out global state on the next date.
    assert np.isclose(
        breadth["A"].loc[dates[shock_index + 1], "global_fraction_negative"],
        0.0,
    )
    # The same crash is visible to B one date later, because A is a peer for B.
    assert breadth["B"].loc[
        dates[shock_index + 1], "global_fraction_negative"
    ] > 0.0
    # It is not visible on the shock date itself: all global features are shifted.
    assert np.isclose(
        breadth["B"].loc[dates[shock_index], "global_fraction_negative"],
        0.0,
    )
    # VIX is excluded from the peer panel despite its deliberately negative path.
    assert np.isclose(
        breadth["A"].loc[dates[shock_index], "global_fraction_negative"],
        0.0,
    )


def test_local_downside_state_is_scale_free_and_causal() -> None:
    dates = pd.date_range("2021-01-01", periods=80, freq="D", tz="UTC")
    returns = 0.006 * np.sin(np.arange(80) / 3.0)
    returns[55:58] -= 0.04
    panel = {"A": _market_frame(returns, dates)}
    config = DownsideGlobalAdmissionConfig(
        window=20,
        drawdown_lookback=20,
        local_volatility_lookback=10,
        semivariance_window=5,
        minimum_peer_markets=2,
    )
    state = build_local_state(panel, config)["A"].dropna()

    assert not state.empty
    assert (state["local_downside_z"] <= 1e-12).all()
    assert (state["local_drawdown_60"] <= 1e-12).all()
    assert state["local_downside_share_5"].between(0.0, 1.0).all()


def test_occurrence_fit_ignores_nonadmitted_nan_rows() -> None:
    rows = 80
    dates = pd.Series(pd.date_range("2010-01-01", periods=rows, freq="D", tz="UTC"))
    labels = np.asarray([index % 2 for index in range(rows)], dtype=int)
    signal = np.where(labels == 1, 1.0, -1.0)
    features = np.column_stack([signal, signal**3])
    features[[3, 77]] = np.nan
    train = np.zeros(rows, dtype=bool)
    train[:60] = True
    validation = ~train
    config = DownsideGlobalAdmissionConfig(
        window=10,
        drawdown_lookback=10,
        local_volatility_lookback=5,
        minimum_peer_markets=2,
    )

    probability, selected, candidates = fit_select_occurrence(
        features,
        labels,
        dates,
        train,
        validation,
        config,
    )

    finite_validation = validation & np.isfinite(features).all(axis=1)
    assert np.isfinite(probability[finite_validation]).all()
    assert np.isnan(probability[77])
    assert float(selected["inner_average_precision"]) > 0.95
    assert len(candidates) == len(config.gate_cs)
