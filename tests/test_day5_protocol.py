from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5 import protocol


def test_add_extrema_preserves_input_and_matches_locked_geometry() -> None:
    frame = pd.DataFrame(
        {
            "r_d1": [-0.03, 0.01],
            "r_d2": [-0.05, 0.02],
            "r_d3": [-0.02, 0.00],
            "r_d4": [0.01, -0.01],
            "r_d5": [0.02, 0.03],
            "current_return_5d_from_branch": [0.02, 0.03],
            "distance_to_relapse_barrier": [0.10, 0.08],
            "distance_to_recovery_barrier": [0.06, 0.07],
        }
    )

    result = protocol.add_extrema(frame)

    assert "closest_to_relapse" not in frame.columns
    np.testing.assert_allclose(result["closest_to_relapse"], [0.03, 0.06])
    np.testing.assert_allclose(result["closest_to_recovery"], [0.07, 0.08])


def test_differential_patterns_are_complementary_pairs() -> None:
    train = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    values = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])

    patterns = protocol.differential_patterns(train, values)

    assert patterns.shape == (1, 10)
    np.testing.assert_allclose(patterns[:, 0::2] + patterns[:, 1::2], 1.0)
    assert np.all((patterns >= 0.0) & (patterns <= 1.0))


def test_eligible_rows_uses_cluster_start_and_two_class_history() -> None:
    dates = pd.date_range("1989-01-01", periods=33, freq="YS")
    frame = pd.DataFrame(
        {
            "landmark_date": dates,
            "cluster_start": dates,
            "y_recovery": [0, 1] * 16 + [1],
        }
    )

    assert protocol.eligible_rows(frame) == [30, 31, 32]


def test_spatial_runner_reexports_protocol_symbols() -> None:
    script = Path("scripts/modeling/run_day5_spatial_rydberg_assay_shard.py")
    spec = importlib.util.spec_from_file_location("day5_spatial_runner", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.load_frame is protocol.load_frame
    assert module.add_extrema is protocol.add_extrema
    assert module.differential_patterns is protocol.differential_patterns
    assert module.eligible_rows is protocol.eligible_rows
    assert module.rydberg_config is protocol.rydberg_config
    assert module.D1 == protocol.D1
    assert module.STATIC == protocol.STATIC
