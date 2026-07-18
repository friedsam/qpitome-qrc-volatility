from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/run_cross_market_range_formula_audit.py")
SPEC = importlib.util.spec_from_file_location("run_cross_market_range_formula_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_range_formulas_contains_expected_variants() -> None:
    dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
    close = np.linspace(100.0, 120.0, 50)
    panel = pd.DataFrame({
        "date": dates,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "ticker": "^GSPC",
    })
    formulas = MODULE.build_range_formulas(panel)["^GSPC"]
    expected = {
        "log_high_low",
        "pct_log_high_low",
        "range_over_low",
        "range_over_close",
        "range_over_mid",
        "sqrt_log_high_low",
        "squared_log_high_low",
        "parkinson_variance",
        "annualized_parkinson_sigma",
    }
    assert expected.issubset(formulas.columns)


def test_per_market_affine_recovers_market_scaling() -> None:
    pairs = pd.DataFrame({
        "market_group": ["a"] * 5 + ["b"] * 5,
        "raw": np.tile(np.arange(1.0, 6.0), 2),
        "stage": np.concatenate([2.0 * np.arange(1.0, 6.0) + 3.0, 5.0 * np.arange(1.0, 6.0) - 4.0]),
    })
    rows = MODULE.evaluate_candidate(pairs, "synthetic")
    per_market = next(row for row in rows if row["normalization"] == "per_market_affine")
    assert per_market["rmse"] < 1e-10
    assert per_market["correlation"] > 0.999999
