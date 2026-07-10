from __future__ import annotations

import numpy as np

from qpitome_qrc.qrc.rydberg_interaction_map import (
    RydbergInteractionConfig,
    classical_ring_products,
    make_rydberg_evolution,
    rydberg_interaction_features,
)


def test_rydberg_evolution_is_unitary() -> None:
    evolution = make_rydberg_evolution(RydbergInteractionConfig())
    identity = np.eye(evolution.shape[0])
    np.testing.assert_allclose(
        evolution.conj().T @ evolution,
        identity,
        atol=1e-10,
    )


def test_rydberg_features_are_six_finite_connected_correlations() -> None:
    x = np.asarray(
        [
            [0.0, 0.5, -0.5, 1.0, -1.0, 0.25],
            [1.0, -1.0, 0.2, -0.2, 0.8, -0.8],
        ],
        dtype=float,
    )
    features = rydberg_interaction_features(x)
    assert features.shape == (2, 6)
    assert np.isfinite(features).all()
    assert np.max(np.abs(features)) <= 0.25 + 1e-10


def test_rydberg_map_is_nonlinear_in_input_configuration() -> None:
    x = np.asarray([[0.8, -0.4, 0.6, -0.2, 0.5, -0.7]], dtype=float)
    y = -x
    midpoint = rydberg_interaction_features(np.zeros_like(x))
    average = 0.5 * (
        rydberg_interaction_features(x)
        + rydberg_interaction_features(y)
    )
    assert not np.allclose(midpoint, average)


def test_classical_ring_products_match_six_feature_budget() -> None:
    x = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=float)
    features = classical_ring_products(x)
    assert features.shape == (1, 6)
    np.testing.assert_allclose(features[0], [2.0, 6.0, 9.0, 9.0, 9.0, 3.0])
