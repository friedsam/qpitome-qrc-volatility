from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.representation_candidates import (
    REPRESENTATIONS,
    CandidateFeatureConfig,
    build_candidate_sequences,
    elementwise_quadratic_matrix,
    fit_channel_scaler,
    transform_candidate_sequences,
)


def test_all_representations_are_causal_and_finite() -> None:
    base = np.asarray(
        [
            [0.0, 0.2, 0.1, 0.4, 0.3, 0.8, 0.7, 1.0],
            [0.3, 0.1, 0.2, 0.0, -0.1, 0.2, 0.1, 0.4],
        ]
    )
    changed = base.copy()
    changed[:, 6:] += 100.0
    config = CandidateFeatureConfig()

    for representation in REPRESENTATIONS:
        original = build_candidate_sequences(base, representation, config)
        perturbed = build_candidate_sequences(changed, representation, config)
        assert original.shape == (2, 8, 2)
        assert np.isfinite(original).all()
        assert np.allclose(original[:, :6], perturbed[:, :6])


def test_level_only_has_constant_second_channel() -> None:
    level = np.asarray([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]])
    sequences = build_candidate_sequences(level, "level_only")
    assert np.allclose(sequences[:, :, 1], 0.0)


def test_scaler_is_train_only_and_bounded() -> None:
    level = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4, 0.5],
            [100.0, 110.0, 120.0, 130.0],
        ]
    )
    sequences = build_candidate_sequences(level, "level_instability")
    train_mask = np.asarray([True, True, False])
    scaler = fit_channel_scaler(sequences, train_mask)
    transformed = transform_candidate_sequences(sequences, scaler)
    assert transformed.shape == sequences.shape
    assert np.max(np.abs(transformed)) <= 1.0
    assert scaler.medians[0] < 1.0


def test_elementwise_quadratic_mirror_has_expected_width() -> None:
    sequences = np.arange(2 * 4 * 2, dtype=float).reshape(2, 4, 2)
    matrix = elementwise_quadratic_matrix(sequences)
    assert matrix.shape == (2, 20)
    level = sequences[:, :, 0]
    second = sequences[:, :, 1]
    first_step = matrix.reshape(2, 4, 5)[:, 0]
    assert np.allclose(first_step[:, 0], level[:, 0])
    assert np.allclose(first_step[:, 1], second[:, 0])
    assert np.allclose(first_step[:, 4], level[:, 0] * second[:, 0])
