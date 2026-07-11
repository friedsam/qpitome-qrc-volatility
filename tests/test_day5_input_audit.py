"""Tests for the classical D1 path-shape input audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_input_audit.py"
SPEC = importlib.util.spec_from_file_location("day5_input_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def toy_frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "r_d1": [1.0, -1.0],
        "r_d2": [2.0, -2.0],
        "r_d3": [1.0, -1.0],
        "r_d4": [3.0, -3.0],
        "r_d5": [4.0, -4.0],
        "current_return_5d_from_branch": [4.0, -4.0],
        "distance_to_recovery_barrier": [2.0, 2.0],
        "distance_to_relapse_barrier": [2.0, 2.0],
        "barrier_width": [4.0, 4.0],
    })
    return frame


def test_path_shape_features_are_finite() -> None:
    out = module.add_path_shape_features(toy_frame())
    columns = module.BLOCKS["all_path_shape"]
    assert np.all(np.isfinite(out[columns].to_numpy(float)))


def test_total_variation_and_reversals() -> None:
    out = module.add_path_shape_features(toy_frame())
    assert out.loc[0, "path_total_variation"] == 8.0
    assert out.loc[0, "path_reversal_count"] == 2
    assert out.loc[1, "path_total_variation"] == 8.0
    assert out.loc[1, "path_reversal_count"] == 2


def test_blocks_are_unique_and_nonempty() -> None:
    for name, columns in module.BLOCKS.items():
        assert columns, name
        assert len(columns) == len(set(columns)), name
    assert len(module.BLOCKS["all_path_shape"]) == 17
