"""Focused tests for the bounded day-5 static nonlinear probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "modeling"
    / "run_cross_market_day5_static_nonlinear_probe.py"
)
SPEC = importlib.util.spec_from_file_location("day5_static_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


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

    lower = 0.015 - 0.055
    upper = 0.015 + 0.035
    assert out.loc[0, "closest_to_relapse"] == (-0.03 - lower)
    assert out.loc[0, "closest_to_recovery"] == (upper - 0.02)


def test_random_tanh_features_are_deterministic_and_train_scaled() -> None:
    train = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    test = np.array([[3.0, 6.0]])

    train_a, test_a = probe.random_tanh_features(train, test, seed=42)
    train_b, test_b = probe.random_tanh_features(train, test, seed=42)

    np.testing.assert_allclose(train_a, train_b)
    np.testing.assert_allclose(test_a, test_b)
    assert train_a.shape == (3, probe.RANDOM_FEATURE_DIM)
    assert test_a.shape == (1, probe.RANDOM_FEATURE_DIM)


def test_paired_delta_uses_exact_common_cohort() -> None:
    predictions = pd.DataFrame(
        {
            "y": [0, 1, 1],
            "D1": [0.2, 0.7, 0.8],
            "candidate": [0.1, np.nan, 0.9],
            "cluster_id": [1, 2, 3],
        }
    )

    result = probe.paired_delta(predictions, "candidate")

    assert result["n"] == 2
    assert result["n_clusters"] == 2
    assert np.isfinite(result["delta_logloss"])
    assert np.isfinite(result["delta_brier"])


def test_independent_static_excludes_redundant_barrier_width() -> None:
    assert "barrier_width" not in probe.INDEPENDENT_STATIC
    assert "distance_to_recovery_barrier" in probe.INDEPENDENT_STATIC
    assert "distance_to_relapse_barrier" in probe.INDEPENDENT_STATIC
