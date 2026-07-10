from __future__ import annotations

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_transition_path import (
    TRANSITION_PATH_COLUMNS,
    add_transition_path_channels,
    extract_transition_windows,
)


def make_daily(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0002, 0.012, size=n)
    price = 100.0 * np.exp(np.cumsum(returns))
    rv20 = np.full(n, 0.20)
    rv5 = 0.20 * np.exp(0.15 * np.sin(np.linspace(0, 8, n)))
    return pd.DataFrame(
        {
            "date": pd.date_range("2000-01-03", periods=n, freq="B"),
            "spy_log_return": returns,
            "rv_5d": rv5,
            "rv_20d": rv20,
            "spy_adj_close": price,
        }
    )


def test_transition_channels_are_causal_under_future_append() -> None:
    daily = make_daily(180)
    cutoff = 150

    short = add_transition_path_channels(daily.iloc[:cutoff].copy())
    full = add_transition_path_channels(daily.copy())

    for column in TRANSITION_PATH_COLUMNS:
        np.testing.assert_allclose(
            short[column].to_numpy(),
            full.loc[: cutoff - 1, column].to_numpy(),
            equal_nan=True,
        )


def test_transition_channels_respect_fixed_clip_bounds() -> None:
    daily = make_daily(180)
    daily.loc[170, "spy_log_return"] = -0.5
    daily.loc[171, "spy_log_return"] = 0.5

    out = add_transition_path_channels(daily)

    assert out["signed_return_over_local_vol"].dropna().between(-4.0, 4.0).all()
    assert out["downside_shock_pressure"].dropna().between(0.0, 1.0).all()
    assert out["d_log_rv5_over_rv20"].dropna().between(-1.0, 1.0).all()
    assert out["drawdown_repair_over_local_vol"].dropna().between(-4.0, 4.0).all()


def test_same_endpoint_can_preserve_different_ordered_paths() -> None:
    daily_a = make_daily(180)
    daily_b = daily_a.copy()

    # Reverse only an interior return segment, preserving its cumulative sum and
    # therefore the segment endpoint price after reconstructing prices.
    segment = daily_b.loc[135:154, "spy_log_return"].to_numpy()[::-1]
    daily_b.loc[135:154, "spy_log_return"] = segment
    daily_b["spy_adj_close"] = 100.0 * np.exp(np.cumsum(daily_b["spy_log_return"]))

    out_a = add_transition_path_channels(daily_a)
    out_b = add_transition_path_channels(daily_b)

    endpoint = 154
    assert np.isclose(
        daily_a.loc[endpoint, "spy_adj_close"],
        daily_b.loc[endpoint, "spy_adj_close"],
    )

    path_a = out_a.loc[135:154, list(TRANSITION_PATH_COLUMNS)].to_numpy()
    path_b = out_b.loc[135:154, list(TRANSITION_PATH_COLUMNS)].to_numpy()
    assert not np.allclose(path_a, path_b, equal_nan=True)


def test_extract_transition_windows_ends_at_branch_index() -> None:
    daily = add_transition_path_channels(make_daily(180))
    episodes = pd.DataFrame(
        {
            "episode_id": [3, 7],
            # The drawdown-repair channel has a 120-row warm-up. Use branch
            # points whose full 40-row windows are beyond that warm-up.
            "branch_idx": [160, 170],
        }
    )

    ids, windows = extract_transition_windows(daily, episodes, lookback=40)

    assert ids.tolist() == [3, 7]
    assert windows.shape == (2, 40, 4)
    expected = daily.loc[131:170, list(TRANSITION_PATH_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(windows[1], expected)
