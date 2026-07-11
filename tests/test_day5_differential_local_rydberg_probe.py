"""Focused tests for the frozen differential local-Rydberg outer probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_cross_market_day5_differential_local_rydberg_probe.py"
SPEC = importlib.util.spec_from_file_location("day5_diff_rydberg_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_differential_patterns_are_complementary() -> None:
    train = np.vstack([np.zeros(5), np.ones(5), 2.0 * np.ones(5)])
    rows = np.array([[0.0, 1.0, -1.0, 2.0, -2.0]])
    patterns = probe.differential_patterns(train, rows)

    assert patterns.shape == (1, 10)
    np.testing.assert_allclose(patterns[:, 0::2] + patterns[:, 1::2], 1.0)
    assert np.min(patterns) >= 0.0
    assert np.max(patterns) <= 1.0


def test_frozen_configuration_matches_diagnostic() -> None:
    config = probe.rydberg_config()
    assert config.reservoir.geometry == "chain"
    assert config.reservoir.chain_atoms == 10
    assert config.reservoir.chain_spacing_um == 7.5
    assert config.reservoir.observable_mode == "n_nn"
    assert config.evolution_time_us == 0.55
    assert config.global_omega_rad_us == 6.0
    assert config.global_delta_rad_us == 6.0
    assert config.local_delta_rad_us == 4.0


def test_frozen_feature_dimension_is_55() -> None:
    train = np.vstack([np.zeros(5), np.ones(5), 2.0 * np.ones(5)])
    patterns = probe.differential_patterns(train, train)
    features = probe.build_local_detuning_feature_matrix(patterns, probe.rydberg_config())

    assert features.shape == (3, 55)
    assert np.all(np.isfinite(features))
