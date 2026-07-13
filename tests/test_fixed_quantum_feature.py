from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from qpitome_qrc.qrc import fixed_quantum_feature as fixed


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_quantum_feature_shape_and_ranges() -> None:
    features = fixed.quantum_features(np.zeros(fixed.N_QUBITS))

    assert features.shape == (14,)
    assert np.isfinite(features).all()
    assert np.all(features >= -1.0 - 1e-12)
    assert np.all(features <= 1.0 + 1e-12)


def test_fixed_quantum_gates_preserve_state_norm_and_clip_inputs() -> None:
    state = np.zeros(fixed.N_STATE, dtype=complex)
    state[0] = 1.0
    state = fixed.apply_ry(state, 0, 0.7)
    state = fixed.apply_rx(state, 1, -0.4)
    state = fixed.apply_zz_phase(state, 0, 1, 0.3)

    np.testing.assert_allclose(np.vdot(state, state).real, 1.0)
    large = fixed.quantum_features(np.full(fixed.N_QUBITS, 100.0))
    clipped = fixed.quantum_features(np.full(fixed.N_QUBITS, fixed.INPUT_CLIP))
    np.testing.assert_allclose(large, clipped)


def test_legacy_fixed_quantum_script_reexports_package_components() -> None:
    module = load_script(
        "legacy_fixed_quantum_feature_map",
        "scripts/modeling/run_cross_market_day5_fixed_quantum_feature.py",
    )

    assert module.quantum_features is fixed.quantum_features
    assert module.apply_ry is fixed.apply_ry
    assert module.apply_rx is fixed.apply_rx
    assert module.apply_zz_phase is fixed.apply_zz_phase
    assert module.expectation_z is fixed.expectation_z
    assert module.expectation_x is fixed.expectation_x
    assert module.expectation_zz is fixed.expectation_zz
    assert module.N_QUBITS == fixed.N_QUBITS
    assert module.N_STATE == fixed.N_STATE
    assert module.N_LAYERS == fixed.N_LAYERS
    assert module.INPUT_CLIP == fixed.INPUT_CLIP
    assert module.ENTANGLING_PAIRS is fixed.ENTANGLING_PAIRS
