from __future__ import annotations

import numpy as np

from transition_forecasting.classical_baselines.comparable_esn import (
    FieldESNSpec,
    _field_sequence,
    _make_weights,
    _pooled_esn_features,
)


def test_field_sequence_uses_train_only_scaling() -> None:
    level = np.arange(4 * 40, dtype=float).reshape(4, 40)
    train = np.array([True, True, False, False])
    sequence = _field_sequence(level, train)

    assert sequence.shape == (4, 40, 3)
    assert np.isclose(sequence[train, :, 0].mean(), 0.0, atol=1e-6)
    assert np.isclose(sequence[train, :, 0].std(), 1.0, atol=1e-6)
    assert np.allclose(sequence[:, 0, 1], 0.0)
    assert np.allclose(sequence[:, :, 2], np.linspace(0.0, 1.0, 40))


def test_field_weights_and_features_are_deterministic() -> None:
    spec = FieldESNSpec(reservoir_size=20)
    w_in_a, recurrent_a = _make_weights(1, spec)
    w_in_b, recurrent_b = _make_weights(1, spec)
    assert np.array_equal(w_in_a, w_in_b)
    assert np.array_equal(recurrent_a.toarray(), recurrent_b.toarray())

    rng = np.random.default_rng(7)
    sequence = rng.normal(size=(6, 40, 3)).astype(np.float32)
    active = np.array([True, True, True, True, False, False])
    features_a = _pooled_esn_features(
        sequence,
        active,
        1,
        shuffled=False,
        spec=spec,
    )
    features_b = _pooled_esn_features(
        sequence,
        active,
        1,
        shuffled=False,
        spec=spec,
    )
    assert features_a.shape == (6, 60)
    assert np.allclose(features_a, features_b)
