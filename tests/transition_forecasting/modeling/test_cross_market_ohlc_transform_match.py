from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/run_cross_market_ohlc_transform_match.py")
SPEC = importlib.util.spec_from_file_location("run_cross_market_ohlc_transform_match", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_ohlc_signals_contains_estimators() -> None:
    dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
    close = np.exp(np.linspace(0.0, 0.2, 50))
    panel = pd.DataFrame({
        "date": dates,
        "open": close * 0.995,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "ticker": "^GSPC",
    })
    signals = MODULE.build_ohlc_signals(panel)["^GSPC"]
    for column in (
        "parkinson",
        "garman_klass",
        "rogers_satchell",
        "true_range_pct",
        "log_parkinson_mean5",
        "garman_klass_rms10",
    ):
        assert column in signals


def test_compare_recovers_exact_candidate_and_alignment() -> None:
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    phase = np.arange(100)
    close = np.exp(np.cumsum(0.002 + 0.01 * np.sin(phase / 7.0)))
    panel = pd.DataFrame({
        "date": dates,
        "open": close * (1.0 - 0.002 * np.cos(phase / 5.0)),
        "high": close * (1.01 + 0.002 * np.sin(phase / 3.0)),
        "low": close * (0.99 - 0.002 * np.cos(phase / 4.0)),
        "close": close,
        "ticker": "^GSPC",
    })
    signals = MODULE.build_ohlc_signals(panel)
    origin = dates[-1]
    target = MODULE.extract_window(
        signals["^GSPC"]["log_parkinson_mean5"], origin, 40, "previous_or_same"
    )
    assert target is not None
    manifest = pd.DataFrame({
        "sample_id": ["s1"],
        "market_group": ["united_states"],
        "origin_date": [origin],
    })
    results = MODULE.compare(manifest, target[None, :], signals)
    exact = results.loc[
        (results["alignment"] == "previous_or_same")
        & (results["correlation"] > 0.999999)
        & (results["affine_rmse"] < 1e-10)
    ]
    assert "log_parkinson_mean5" in set(exact["candidate"])
