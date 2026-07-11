"""Tests for the reusable day-5 feature assay."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/modeling/run_day5_standard_feature_assay.py"
SPEC = importlib.util.spec_from_file_location("day5_standard_feature_assay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_leave_one_out_spec_excludes_tested_feature() -> None:
    spec = json.loads((REPO / "configs/day5_feature_tests/d1_leave_one_out.json").read_text())
    assert len(spec["blocks"]) == 4
    for name, block in spec["blocks"].items():
        assert block["features"] == [name]
        assert name not in block["baseline_features"]
        assert len(block["baseline_features"]) == 3


def test_path_shape_spec_uses_full_d1_baseline() -> None:
    spec = json.loads((REPO / "configs/day5_feature_tests/path_shape_selected.json").read_text())
    assert spec["preparation"] == "path_shape"
    expected = set(module.base.assay.D1)
    for block in spec["blocks"].values():
        assert set(block["baseline_features"]) == expected


def test_nonlinear_maps_are_deterministic_and_shaped() -> None:
    train = np.array([[0.0], [1.0], [2.0], [3.0]])
    test = np.array([[1.5], [2.5]])
    first = module.nonlinear_maps(train, test, seed=5)
    second = module.nonlinear_maps(train, test, seed=5)
    assert first["poly2"][0].shape == (4, 2)
    assert first["tanh32"][0].shape == (4, module.TANH_WIDTH)
    assert np.allclose(first["tanh32"][0], second["tanh32"][0])


def test_score_delta_zero_for_identical_predictions() -> None:
    frame = pd.DataFrame({
        "y": [0, 1, 0, 1],
        "baseline": [0.2, 0.8, 0.3, 0.7],
        "model": [0.2, 0.8, 0.3, 0.7],
        "cluster_id": ["a", "a", "b", "b"],
    })
    result = module.score_deltas(frame, "baseline", "model")
    assert np.isclose(result["delta_logloss"], 0.0)
    assert np.isclose(result["cluster_mean_delta_logloss"], 0.0)
