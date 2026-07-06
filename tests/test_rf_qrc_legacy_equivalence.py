"""Regression tests for the Phase 3 RF-QRC extraction.

The legacy experiment script remains untouched while these tests compare its
embedded implementation directly against the reusable source module.  This is
the safety net for subsequent runner cleanup.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qpitome_qrc.qrc import rf_qrc_reservoir as extracted


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = REPO_ROOT / "scripts" / "run_phase3_rf_qrc_tail_probe.py"


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_rf_qrc_tail_probe", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy script: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def legacy():
    return load_legacy_module()


@pytest.mark.parametrize("theta", [-2.1, -0.3, 0.0, 0.7, 2.4])
def test_single_qubit_gates_match_legacy(legacy, theta: float) -> None:
    np.testing.assert_allclose(extracted.ry(theta), legacy.ry(theta), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(extracted.rz(theta), legacy.rz(theta), rtol=0.0, atol=0.0)


def test_low_level_state_operations_match_legacy(legacy) -> None:
    rng = np.random.default_rng(7)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    state = state / np.linalg.norm(state)

    gate = extracted.ry(0.37)
    np.testing.assert_allclose(
        extracted.apply_one(state, gate, 1, 3),
        legacy.apply_one(state, gate, 1, 3),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        extracted.apply_cnot(state, 0, 2, 3),
        legacy.apply_cnot(state, 0, 2, 3),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        extracted.apply_zz_phase(state, 0, 2, 0.41, 3),
        legacy.apply_zz_phase(state, 0, 2, 0.41, 3),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        extracted.z_zz_features(state, 3),
        legacy.z_zz_features(state, 3),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("second_encoding", "entangler"),
    [
        (False, "all_pairs"),
        (True, "all_pairs"),
        (True, "ring"),
    ],
)
def test_feature_map_matches_legacy(
    legacy,
    second_encoding: bool,
    entangler: str,
) -> None:
    rng = np.random.default_rng(123)
    inputs = rng.normal(size=(7, 6))
    args = (6, second_encoding, entangler, np.pi / 3, 42)

    old_map = legacy.RFQRCMap(*args)
    new_map = extracted.RFQRCMap(*args)

    np.testing.assert_allclose(new_map.rz_angles, old_map.rz_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_map.ry_angles, old_map.ry_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_map.zz_angles, old_map.zz_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        new_map.transform(inputs),
        old_map.transform(inputs),
        rtol=1e-13,
        atol=1e-13,
    )


def test_state_norm_and_feature_dimension() -> None:
    model = extracted.RFQRCMap(6, True, "ring", np.pi / 3, 42)
    u = np.linspace(-1.5, 1.5, 6)

    state = model.state(u)
    features = model.one(u)

    np.testing.assert_allclose(np.linalg.norm(state), 1.0, rtol=1e-13, atol=1e-13)
    assert features.shape == (model.n_features,)
    assert model.n_features == 21


def test_transform_is_deterministic() -> None:
    inputs = np.arange(24, dtype=float).reshape(4, 6) / 10.0
    first = extracted.RFQRCMap(6, True, "ring", np.pi / 3, 42).transform(inputs)
    second = extracted.RFQRCMap(6, True, "ring", np.pi / 3, 42).transform(inputs)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_entangler_topologies_are_explicit() -> None:
    assert extracted.entangler_pairs("none", 4) == []
    assert extracted.entangler_pairs("all_pairs", 4) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert extracted.entangler_pairs("ring", 4) == [(0, 1), (1, 2), (2, 3), (3, 0)]
