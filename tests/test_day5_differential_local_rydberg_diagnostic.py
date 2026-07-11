"""Focused tests for differential-pair local-detuning encoding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "modeling"
    / "run_day5_differential_local_rydberg_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("differential_rydberg_diag", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_complementary_pairs_sum_to_one() -> None:
    z = np.array([[0.0, 1.0, -1.0, 2.0, -2.0]])
    patterns = probe.differential_pair_patterns(z)

    assert patterns.shape == (1, 10)
    np.testing.assert_allclose(patterns[:, 0::2] + patterns[:, 1::2], 1.0)


def test_sign_swap_exchanges_pair_members() -> None:
    z = np.array([[0.5, -0.7, 1.2, -1.5, 0.2]])
    positive = probe.differential_pair_patterns(z)
    negative = probe.differential_pair_patterns(-z)

    np.testing.assert_allclose(positive[:, 0::2], negative[:, 1::2])
    np.testing.assert_allclose(positive[:, 1::2], negative[:, 0::2])


def test_diagnostic_feature_dimension_is_55() -> None:
    config = probe.diagnostic_config()
    patterns = probe.differential_pair_patterns(np.zeros((2, 5)))
    features = probe.build_local_detuning_feature_matrix(patterns, config)

    assert features.shape == (2, 55)
    assert np.all(np.isfinite(features))


def test_targets_include_branch_motivated_structures() -> None:
    z = np.zeros((4, 5))
    target_map = probe.targets(z)

    assert "extrema_asymmetry" in target_map
    assert "endpoint_barrier_interaction" in target_map
    assert "closest_barrier_softmin" in target_map
    assert "mixed_state_score" in target_map


def test_effective_rank_is_bounded_by_feature_count() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(30, 7))
    rank = probe.effective_rank(features)

    assert 1.0 <= rank["participation_ratio"] <= 7.0
    assert 1 <= rank["n95"] <= 7
