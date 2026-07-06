"""Equivalence tests for the qubit-scaling selected-observable RF-QRC map."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qpitome_qrc.qrc.rf_qrc_observables import (
    SelectedObservableRFQRCMap,
    selected_z_zz_features,
    selected_zz_pairs,
)


LEGACY_REFERENCE = Path(__file__).with_name("legacy_qubit_scaling_reference.py")


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_qubit_scaling", LEGACY_REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy reference: {LEGACY_REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("zz_mode", ["ring", "ring_plus_next", "all"])
def test_selected_observable_map_matches_legacy(zz_mode: str) -> None:
    legacy = load_legacy_module()
    rng = np.random.default_rng(2026)
    inputs = rng.normal(size=(5, 8))

    old_map = legacy.RFQRCRingMap(8, np.pi / 3, 0.35, 42, zz_mode)
    new_map = SelectedObservableRFQRCMap(
        n_qubits=8,
        input_scale=np.pi / 3,
        random_scale=0.35,
        seed=42,
        zz_mode=zz_mode,
    )

    np.testing.assert_allclose(new_map.rz_angles, old_map.rz_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_map.ry_angles, old_map.ry_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_map.zz_angles, old_map.zz_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        new_map.transform(inputs),
        old_map.transform(inputs),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("zz_mode", ["ring", "ring_plus_next", "all"])
def test_selected_feature_function_matches_legacy(zz_mode: str) -> None:
    legacy = load_legacy_module()
    rng = np.random.default_rng(11)
    state = rng.normal(size=2**6) + 1j * rng.normal(size=2**6)
    state = state / np.linalg.norm(state)

    expected = legacy.selected_z_zz_features(state, 6, zz_mode)
    actual = selected_z_zz_features(state, 6, zz_mode)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_selected_pair_counts() -> None:
    assert len(selected_zz_pairs(6, "ring")) == 6
    assert len(selected_zz_pairs(6, "ring_plus_next")) == 12
    assert len(selected_zz_pairs(6, "all")) == 15
