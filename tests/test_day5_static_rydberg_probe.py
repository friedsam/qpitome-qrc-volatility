"""Focused tests for the state-conditioned day-5 Rydberg probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_cross_market_day5_static_rydberg_probe.py"
SPEC = importlib.util.spec_from_file_location("day5_static_rydberg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_masks_are_fixed_and_have_expected_shape() -> None:
    first = probe.make_masks()
    second = probe.make_masks()
    assert first.shape == (probe.MASK_COUNT, 2, len(probe.STATIC))
    np.testing.assert_allclose(first, second)


def test_mask_windows_use_train_scaling_and_stay_bounded() -> None:
    train = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ]
    )
    all_rows = np.vstack([train, np.array([[100.0, 100.0, 100.0, 100.0, 100.0]])])
    windows = probe.make_mask_windows(train, all_rows, probe.make_masks())

    assert windows.shape == (4, probe.MASK_COUNT, 2)
    assert np.all(np.isfinite(windows))
    assert np.max(np.abs(windows)) <= 1.0


def test_rydberg_configuration_matches_protocol() -> None:
    config = probe.rydberg_config()
    assert config.memory_mode == "memoryless"
    assert config.observable_mode == "n"
    assert config.lookback_days == probe.MASK_COUNT
    assert config.anchor_count == probe.MASK_COUNT
    assert config.total_time_us / probe.MASK_COUNT == probe.MASK_EVOLUTION_US
    assert config.shots is None


def test_expected_rydberg_feature_dimension() -> None:
    config = probe.rydberg_config()
    windows = np.zeros((2, probe.MASK_COUNT, 2), dtype=float)
    features = probe.build_rydberg_feature_matrix(windows, config)

    assert features.shape == (2, probe.MASK_COUNT * 8)
    assert np.all(np.isfinite(features))


def test_add_extrema_matches_manual_geometry() -> None:
    frame = pd.DataFrame(
        {
            "r_d1": [-0.03],
            "r_d2": [-0.01],
            "r_d3": [0.02],
            "r_d4": [0.01],
            "r_d5": [0.015],
            "current_return_5d_from_branch": [0.015],
            "distance_to_relapse_barrier": [0.055],
            "distance_to_recovery_barrier": [0.035],
        }
    )
    out = probe.add_extrema(frame)
    assert out.loc[0, "closest_to_relapse"] == (-0.03 - (0.015 - 0.055))
    assert out.loc[0, "closest_to_recovery"] == ((0.015 + 0.035) - 0.02)
