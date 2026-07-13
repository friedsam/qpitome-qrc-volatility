from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from qpitome_qrc.qrc import fixed_rydberg_feature as fixed


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_rydberg_feature_shape_and_observable_ranges() -> None:
    features = fixed.rydberg_features(np.zeros(fixed.N_QUBITS))

    assert features.shape == (12,)
    assert np.isfinite(features).all()
    assert np.all((features[:10] >= 0.0) & (features[:10] <= 1.0))
    assert features[10] >= 0.0
    np.testing.assert_allclose(features[11], features[:4].sum())


def test_fixed_rydberg_feature_clips_inputs() -> None:
    large = fixed.rydberg_features(np.full(fixed.N_QUBITS, 100.0))
    clipped = fixed.rydberg_features(np.full(fixed.N_QUBITS, fixed.INPUT_CLIP))

    np.testing.assert_allclose(large, clipped)


def test_legacy_fixed_rydberg_script_reexports_package_feature_map() -> None:
    module = load_script(
        "legacy_fixed_rydberg_feature_map",
        "scripts/modeling/run_cross_market_day5_fixed_rydberg_feature.py",
    )

    assert module.rydberg_features is fixed.rydberg_features
    assert module.build_operators is fixed.build_operators
    assert module.bit_value is fixed.bit_value
    assert module.X_OPS is fixed.X_OPS
    assert module.N_OPS is fixed.N_OPS
    assert module.NN_OPS is fixed.NN_OPS
    assert module.PAIR_V is fixed.PAIR_V
    assert module.N_QUBITS == fixed.N_QUBITS
    assert module.N_STATE == fixed.N_STATE
    assert module.OMEGA == fixed.OMEGA
    assert module.EVOLVE_TIME == fixed.EVOLVE_TIME
    assert module.POSITIONS is fixed.POSITIONS
