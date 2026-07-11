"""Tests for the Rydberg-output PCA reconstruction diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_rydberg_pca_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("day5_pca_reconstruction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_fit_predict_ridge_recovers_simple_mapping() -> None:
    x = np.linspace(-2.0, 2.0, 40)[:, None]
    y = np.column_stack([2.0 * x[:, 0] + 1.0, -x[:, 0] + 0.5])
    prediction = module.fit_predict_ridge(x, y, np.array([[0.25]]), alpha=1e-8)
    np.testing.assert_allclose(prediction[0], [1.5, 0.25], atol=1e-6)


def test_safe_r2_handles_constant_target() -> None:
    assert np.isnan(module.safe_r2(np.ones(5), np.ones(5)))


def test_summarize_reports_componentwise_r2() -> None:
    import pandas as pd

    frame = pd.DataFrame({
        "block": ["occupations"] * 4,
        "mode": ["cumulative"] * 4,
        "component_count": [1] * 4,
        "target": ["D1_logit"] * 4,
        "y_true": [0.0, 1.0, 2.0, 3.0],
        "y_pred": [0.0, 1.0, 2.0, 3.0],
    })
    summary = module.summarize(frame)
    assert len(summary) == 1
    assert summary.iloc[0]["r2"] == 1.0
    assert summary.iloc[0]["rmse"] == 0.0
