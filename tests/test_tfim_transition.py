from __future__ import annotations

import numpy as np

from qpitome_qrc.qrc.tfim_transition import (
    TFIMTransitionConfig,
    make_observables,
    make_tfim_evolution,
    scale_transition_windows,
    tfim_transition_features,
)


def make_windows() -> np.ndarray:
    time = np.linspace(-1.0, 1.0, 40)
    return np.stack(
        [
            np.column_stack(
                [
                    2.0 * np.sin(time),
                    0.5 + 0.2 * np.cos(time),
                    0.3 * np.sin(2.0 * time),
                    1.5 * time,
                ]
            ),
            np.column_stack(
                [
                    2.0 * np.cos(time),
                    0.4 + 0.1 * np.sin(time),
                    -0.2 * np.cos(3.0 * time),
                    -1.0 * time,
                ]
            ),
        ]
    )


def test_fixed_scaling_maps_into_unit_interval() -> None:
    scaled = scale_transition_windows(make_windows())
    assert scaled.shape == (2, 40, 4)
    assert np.max(scaled) <= 1.0
    assert np.min(scaled) >= -1.0


def test_tfim_evolution_is_unitary() -> None:
    evolution = make_tfim_evolution(TFIMTransitionConfig())
    identity = np.eye(evolution.shape[0])
    np.testing.assert_allclose(
        evolution.conj().T @ evolution,
        identity,
        atol=1e-10,
    )


def test_observable_budget_is_eight() -> None:
    observables = make_observables(TFIMTransitionConfig())
    assert len(observables) == 8
    for observable in observables:
        np.testing.assert_allclose(observable, observable.conj().T)


def test_tfim_features_are_finite_and_bounded() -> None:
    features = tfim_transition_features(make_windows())
    assert features.shape == (2, 8)
    assert np.isfinite(features).all()
    assert np.max(np.abs(features)) <= 1.0 + 1e-10


def test_temporal_order_changes_tfim_features() -> None:
    windows = make_windows()
    reversed_windows = windows[:, ::-1, :]
    ordered = tfim_transition_features(windows)
    reversed_features = tfim_transition_features(reversed_windows)
    assert not np.allclose(ordered, reversed_features)
