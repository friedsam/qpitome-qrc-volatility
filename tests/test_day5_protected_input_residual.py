"""Tests for protected residual path-input audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_protected_input_residual.py"
SPEC = importlib.util.spec_from_file_location("day5_protected_input", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_feature_blocks_are_fixed_and_unique() -> None:
    assert list(module.FEATURE_BLOCKS) == [
        "path_efficiency",
        "path_reversal_count",
        "path_early_late_imbalance",
        "trajectory_r_d1",
        "compact_four",
    ]
    assert len(module.FEATURE_BLOCKS["compact_four"]) == 4
    assert len(set(module.FEATURE_BLOCKS["compact_four"])) == 4


def test_score_deltas_returns_zero_for_identical_predictions() -> None:
    frame = pd.DataFrame({
        "y": [0, 1, 0, 1],
        "D1": [0.2, 0.8, 0.4, 0.6],
        "same": [0.2, 0.8, 0.4, 0.6],
        "cluster_id": ["a", "a", "b", "b"],
    })
    result = module.score_deltas(frame, "same")
    assert np.isclose(result["delta_logloss"], 0.0)
    assert np.isclose(result["delta_brier"], 0.0)
    assert np.isclose(result["cluster_mean_delta_logloss"], 0.0)
    assert np.isclose(result["cluster_mean_delta_brier"], 0.0)


def test_fixed_hyperparameters_match_confirmation_protocol() -> None:
    assert module.RIDGE_ALPHA == 10.0
    assert module.OFFSET_L2 == 100.0
