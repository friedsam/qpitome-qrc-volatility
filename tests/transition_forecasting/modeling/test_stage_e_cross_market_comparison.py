from __future__ import annotations

import numpy as np

from transition_forecasting.modeling import cross_market_comparison as MODULE


def test_qlike_zero_for_perfect_prediction() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert MODULE.qlike(y, y) == 0.0


def test_shuffled_control_preserves_own_and_time_channels() -> None:
    x = np.arange(2 * 5 * 9, dtype=float).reshape(2, 5, 9)
    shuffled = MODULE.shuffled_cross_market(x, seed=7)
    assert np.array_equal(shuffled[:, :, :2], x[:, :, :2])
    assert np.array_equal(shuffled[:, :, 8], x[:, :, 8])
    assert not np.array_equal(shuffled[:, :, 2:8], x[:, :, 2:8])


def test_pooled_esn_features_are_deterministic_and_well_shaped() -> None:
    rng = np.random.default_rng(3)
    sequences = rng.normal(size=(4, 12, 3))
    a = MODULE.pooled_esn_features(sequences, seed=11, reservoir_size=20, washout=2)
    b = MODULE.pooled_esn_features(sequences, seed=11, reservoir_size=20, washout=2)
    assert a.shape == (4, 60)
    assert np.allclose(a, b)
