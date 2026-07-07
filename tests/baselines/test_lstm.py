"""Unit tests for the reusable LSTM baseline mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from qpitome_qrc.baselines.lstm import (
    LSTMConfig,
    build_sequences,
    fit_lstm,
    predict_lstm,
)


def test_build_sequences_alignment() -> None:
    features = np.arange(12, dtype=float).reshape(6, 2)
    targets = np.arange(6, dtype=float)

    sequences, aligned_targets = build_sequences(features, targets, lookback=3)

    assert sequences.shape == (4, 3, 2)
    np.testing.assert_array_equal(sequences[0], features[:3])
    np.testing.assert_array_equal(sequences[-1], features[-3:])
    np.testing.assert_array_equal(aligned_targets, np.array([2, 3, 4, 5]))


def test_build_sequences_short_input_returns_empty_arrays() -> None:
    features = np.ones((2, 3), dtype=float)
    targets = np.ones(2, dtype=float)

    sequences, aligned_targets = build_sequences(features, targets, lookback=4)

    assert sequences.shape == (0, 4, 3)
    assert aligned_targets.shape == (0,)


def test_build_sequences_rejects_nonfinite_data() -> None:
    features = np.array([[1.0], [np.nan]])
    targets = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="non-finite"):
        build_sequences(features, targets, lookback=1)


def test_small_lstm_fit_and_predict_is_deterministic() -> None:
    pytest.importorskip("torch")

    rng = np.random.default_rng(7)
    features = rng.normal(size=(24, 2))
    targets = features[:, 0] - 0.5 * features[:, 1]
    sequences, aligned_targets = build_sequences(features, targets, lookback=4)
    config = LSTMConfig(
        hidden_size=4,
        dropout=0.0,
        epochs=2,
        batch_size=8,
        seed=11,
    )

    first = fit_lstm(sequences, aligned_targets, config=config)
    second = fit_lstm(sequences, aligned_targets, config=config)

    first_predictions = predict_lstm(first.model, sequences[:3])
    second_predictions = predict_lstm(second.model, sequences[:3])

    np.testing.assert_allclose(first_predictions, second_predictions, atol=1e-7)
    assert np.isfinite(first.final_train_mse)
