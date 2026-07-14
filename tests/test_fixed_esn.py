from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines import fixed_esn


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_esn_weights_are_deterministic_and_radius_locked() -> None:
    first = fixed_esn.fixed_esn_weights(4)
    second = fixed_esn.fixed_esn_weights(4)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left, right)

    _, reservoir, _ = first
    radius = np.max(np.abs(np.linalg.eigvals(reservoir)))
    np.testing.assert_allclose(radius, fixed_esn.SPECTRAL_RADIUS)


def test_fit_esn_predict_matches_locked_manual_sequence() -> None:
    train_sequences = np.array([
        [[-2.0, 0.0], [-1.0, 0.5], [0.0, 1.0]],
        [[-1.5, 0.2], [-0.5, 0.7], [0.5, 1.2]],
        [[-1.0, -0.4], [0.0, 0.1], [1.0, 0.6]],
        [[0.5, -1.0], [1.0, -0.5], [1.5, 0.0]],
        [[1.0, -0.8], [1.5, -0.3], [2.0, 0.2]],
        [[1.5, -0.6], [2.0, -0.1], [2.5, 0.4]],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    test_sequence = np.array([[0.25, -0.7], [0.75, -0.2], [1.25, 0.3]])
    win, wres, bias = fixed_esn.fixed_esn_weights(2, n_reservoir=8, seed=7)

    actual = fixed_esn.fit_esn_predict(
        train_sequences,
        y,
        test_sequence,
        win,
        wres,
        bias,
        C=0.1,
    )

    scaler = StandardScaler()
    scaler.fit(train_sequences.reshape(-1, train_sequences.shape[-1]))
    train_states = np.vstack([
        fixed_esn.esn_state(scaler.transform(sequence), win, wres, bias)
        for sequence in train_sequences
    ])
    test_state = np.vstack([
        fixed_esn.esn_state(scaler.transform(test_sequence), win, wres, bias)
    ])
    model = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs")
    model.fit(train_states, y)
    expected = float(model.predict_proba(test_state)[0, 1])

    np.testing.assert_allclose(actual, expected)


def test_legacy_fixed_esn_script_reexports_package_core() -> None:
    module = load_script(
        "legacy_fixed_esn",
        "scripts/modeling/day5_branching/temporal_controls/run_cross_market_day5_fixed_esn.py",
    )

    assert module.fixed_esn_weights is fixed_esn.fixed_esn_weights
    assert module.esn_state is fixed_esn.esn_state
    assert module.fit_esn_predict is fixed_esn.fit_esn_predict
    expected_weights = fixed_esn.fixed_esn_weights(4)
    np.testing.assert_allclose(module.WIN, expected_weights[0])
    np.testing.assert_allclose(module.WRES, expected_weights[1])
    np.testing.assert_allclose(module.BIAS, expected_weights[2])
