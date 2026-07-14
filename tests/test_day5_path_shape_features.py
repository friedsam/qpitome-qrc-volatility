from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5.features import PATH_SHAPE_BLOCKS, add_path_shape_features


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_path_shape_features_matches_locked_geometry() -> None:
    frame = pd.DataFrame({
        "r_d1": [0.10],
        "r_d2": [-0.10],
        "r_d3": [0.20],
        "r_d4": [0.10],
        "r_d5": [0.30],
        "current_return_5d_from_branch": [0.30],
        "distance_to_relapse_barrier": [0.50],
        "distance_to_recovery_barrier": [0.20],
        "barrier_width": [0.70],
    })

    result = add_path_shape_features(frame)

    assert "path_total_variation" not in frame.columns
    np.testing.assert_allclose(result["path_total_variation"], [0.90])
    np.testing.assert_allclose(result["path_efficiency"], [1.0 / 3.0])
    np.testing.assert_allclose(result["path_curvature_l1"], [1.50])
    assert int(result.loc[0, "path_reversal_count"]) == 4
    assert int(result.loc[0, "path_argmin_day"]) == 2
    assert int(result.loc[0, "path_argmax_day"]) == 5
    np.testing.assert_allclose(
        result["path_early_late_imbalance"],
        [0.00],
        atol=1e-15,
    )
    assert result.loc[0, "trajectory_r_d4"] == 0.10


def test_input_audit_reexports_path_shape_helpers() -> None:
    module = load_script(
        "day5_input_audit_path_shape",
        "scripts/modeling/day5_branching/falsification/run_day5_input_audit.py",
    )

    assert module.add_path_shape_features is add_path_shape_features
    assert module.BLOCKS is PATH_SHAPE_BLOCKS
