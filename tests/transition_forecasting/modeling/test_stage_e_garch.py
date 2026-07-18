from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.garch import variance_path_to_log_volatility_path

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_garch.py")
SPEC = importlib.util.spec_from_file_location("stage_e_garch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_variance_path_to_log_volatility_restores_return_scale() -> None:
    variance = np.array([1.0, 4.0])
    result = variance_path_to_log_volatility_path(variance, return_scale=100.0)
    expected = np.log(np.array([0.01, 0.02]))
    assert np.allclose(result, expected)


def test_return_history_is_causal_and_bounded() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    close = pd.Series(np.exp(np.arange(8) * 0.01), index=dates)
    result = MODULE._return_history(close, dates[5], history=3)
    assert len(result) == 3
    assert np.allclose(result, 0.01)


def test_load_close_series_normalizes_columns(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-01", "2020-01-02"],
            "Close": [102.0, 100.0, 103.0],
        }
    ).to_csv(path, index=False)
    result = MODULE._load_close_series(path)
    assert list(result.index) == list(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert result.iloc[-1] == 103.0
