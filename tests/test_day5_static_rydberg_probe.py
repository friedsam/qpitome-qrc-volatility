"""Focused tests for the local-detuning day-5 Rydberg probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5.protocol import (
    D1,
    EVAL_START,
    MIN_TRAIN,
    STATIC,
    add_extrema,
    load_frame,
)
from qpitome_qrc.evaluation.binary import logistic_pipeline
from qpitome_qrc.evaluation.scoring import binary_summary, cluster_weighted_summary
from qpitome_qrc.qrc.local_detuning_reservoir import (
    LocalDetuningConfig,
    build_local_detuning_feature_matrix,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_cross_market_day5_static_rydberg_probe.py"
SPEC = importlib.util.spec_from_file_location("day5_static_rydberg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_static_to_local_patterns_shape_bounds_and_train_scaling() -> None:
    train = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ]
    )
    rows = np.vstack([train, np.array([[100.0, 100.0, 100.0, 100.0, 100.0]])])
    patterns = probe.static_to_local_patterns(train, rows)

    assert patterns.shape == (4, 8)
    assert np.all(np.isfinite(patterns))
    assert np.min(patterns) >= 0.0
    assert np.max(patterns) <= 1.0


def test_direct_sites_encode_the_five_coordinates_monotonically() -> None:
    train = np.vstack([np.zeros(5), np.ones(5), 2.0 * np.ones(5)])
    low = np.zeros((1, 5))
    high = 2.0 * np.ones((1, 5))
    patterns = probe.static_to_local_patterns(train, np.vstack([low, high]))

    assert np.all(patterns[1, :5] > patterns[0, :5])


def test_local_detuning_feature_dimension_and_finiteness() -> None:
    patterns = np.full((3, 8), 0.5)
    features = build_local_detuning_feature_matrix(patterns, LocalDetuningConfig())

    assert features.shape == (3, 36)
    assert np.all(np.isfinite(features))


def test_uniform_zero_local_amplitude_ignores_pattern() -> None:
    config = LocalDetuningConfig(local_delta_rad_us=0.0)
    patterns = np.vstack([np.zeros(8), np.ones(8)])
    features = build_local_detuning_feature_matrix(patterns, config)

    np.testing.assert_allclose(features[0], features[1], atol=1e-12, rtol=0)


def test_distinct_local_patterns_change_interacting_features() -> None:
    patterns = np.vstack(
        [
            np.zeros(8),
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ]
    )
    features = build_local_detuning_feature_matrix(patterns, LocalDetuningConfig())

    assert not np.allclose(features[0], features[1], atol=1e-8)


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


def test_static_rydberg_probe_uses_package_helpers_directly() -> None:
    assert probe.D1 == D1
    assert probe.STATIC == STATIC
    assert probe.MIN_TRAIN == MIN_TRAIN
    assert probe.EVAL_START == EVAL_START
    assert probe.add_extrema is add_extrema
    assert probe.load_frame is load_frame
    assert probe.logistic_pipeline is logistic_pipeline
    assert probe.binary_summary is binary_summary
    assert probe.cluster_weighted_summary is cluster_weighted_summary
