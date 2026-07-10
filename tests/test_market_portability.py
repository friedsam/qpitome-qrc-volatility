from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.market_portability import (
    MarketSpec,
    build_detector_input,
    load_market_close,
    summarize_portability,
)


def test_load_market_close_strips_headers_sorts_and_drops_missing(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            " Date ": ["01/03/24", "01/02/24", "01/01/24", "01/04/24"],
            " Close ": [103.0, 102.0, 101.0, "ND"],
        }
    ).to_csv(path, index=False)
    spec = MarketSpec(
        market="Test",
        source="test",
        date_column="Date",
        close_column="Close",
        date_format="%m/%d/%y",
    )

    loaded = load_market_close(path, spec)

    assert loaded["date"].is_monotonic_increasing
    assert loaded["close"].tolist() == [101.0, 102.0, 103.0]


def test_build_detector_input_annualizes_trailing_realized_volatility() -> None:
    close = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30, freq="D"),
            "close": np.exp(np.arange(30) * 0.01),
        }
    )

    daily = build_detector_input(close)

    assert {"spy_adj_close", "spy_log_return", "rv_5d", "rv_20d"}.issubset(daily.columns)
    assert np.isclose(daily["spy_log_return"].dropna().iloc[0], 0.01)
    assert daily["rv_20d"].dropna().max() < 1e-10


def test_summarize_portability_excludes_end_censored_episodes() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=300, freq="D"),
            "spy_adj_close": np.linspace(100.0, 120.0, 300),
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": [1, 2, 3],
            "branch_idx": [50, 100, 250],
        }
    )
    events = pd.DataFrame(
        {
            "episode_id": [1, 2, 3],
            "resolved_within_followup": [True, False, False],
            "event_type": ["recovery", None, None],
            "event_day": [20.0, np.nan, np.nan],
        }
    )
    spec = MarketSpec("Test", "test", "Date", "Close")

    summary = summarize_portability(
        "test",
        spec,
        daily,
        episodes,
        events,
        max_followup=120,
    ).iloc[0]

    assert summary["n_branch_episodes"] == 3
    assert summary["n_complete_120d"] == 2
    assert summary["n_recovery"] == 1
    assert summary["n_unresolved_120d"] == 1
    assert summary["fraction_resolved_120d"] == 0.5
