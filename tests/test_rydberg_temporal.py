from __future__ import annotations

import numpy as np

from qpitome_qrc.qrc.rydberg_temporal import (
    RydbergTemporalConfig,
    make_rydberg_temporal_evolution,
    rydberg_temporal_features,
)


def make_windows() -> np.ndarray:
    time = np.linspace(-1.0, 1.0, 40)
    return np.stack(
        [
            np.column_stack(
                [
                    2.0 * np.sin(time),
                    0.4 + 0.2 * np.cos(time),
                    0.3 * np.sin(2.0 * time),
                    1.2 * time,
                ]
            ),
            np.column_stack(
                [
                    1.5 * np.cos(time),
                    0.3 + 0.1 * np.sin(time),
                    -0.2 * np.cos(3.0 * time),
                    -0.8 * time,
                ]
            ),
        ]
    )


def test_rydberg_temporal_evolution_is_unitary() -> None:
    evolution = make_rydberg_temporal_evolution(RydbergTemporalConfig())
    identity = np.eye(evolution.shape[0])
    np.testing.assert_allclose(evolution.conj().T @ evolution, identity, atol=1e-10)


def test_rydberg_temporal_features_are_eight_and_finite() -> None:
    features = rydberg_temporal_features(make_windows())
    assert features.shape == (2, 8)
    assert np.isfinite(features).all()


def test_temporal_order_changes_rydberg_features() -> None:
    windows = make_windows()
    ordered = rydberg_temporal_features(windows)
    reversed_features = rydberg_temporal_features(windows[:, ::-1, :])
    assert not np.allclose(ordered, reversed_features)
