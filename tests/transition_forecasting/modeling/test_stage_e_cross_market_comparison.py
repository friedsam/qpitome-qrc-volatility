from __future__ import annotations

import numpy as np

from transition_forecasting.modeling import cross_market_comparison as MODULE


def test_metric_pair_zero_for_perfect_prediction() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    qlike, rmse = MODULE.metric_pair(y, y)
    assert qlike == 0.0
    assert rmse == 0.0


def test_hybrid_uses_original_stage_d_channels_not_reconstructed_own_channels() -> None:
    original = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
    compact = np.arange(2 * 5 * 9, dtype=float).reshape(2, 5, 9) + 1000.0
    hybrid = MODULE.build_hybrid_sequences(original, compact)
    assert hybrid.shape == (2, 5, 9)
    assert np.array_equal(hybrid[:, :, :3], original)
    assert np.array_equal(hybrid[:, :, 3:], compact[:, :, 2:8])


def test_shuffled_control_preserves_original_channels() -> None:
    x = np.arange(2 * 8 * 9, dtype=float).reshape(2, 8, 9)
    shuffled = MODULE.shuffle_cross_market_channels(x, seed=7)
    assert np.array_equal(shuffled[:, :, :3], x[:, :, :3])
    assert not np.array_equal(shuffled[:, :, 3:9], x[:, :, 3:9])


def test_original_enrichment_adds_difference_and_time() -> None:
    original = np.array([[[1.0], [3.0], [6.0]]])
    enriched = MODULE.build_original_enriched_sequences(original)
    assert enriched.shape == (1, 3, 3)
    assert np.array_equal(enriched[0, :, 0], [1.0, 3.0, 6.0])
    assert np.array_equal(enriched[0, :, 1], [0.0, 2.0, 3.0])
    assert np.allclose(enriched[0, :, 2], [0.0, 0.5, 1.0])


def test_ridge_scoring_ignores_nan_rows_outside_train_and_validation() -> None:
    features = np.array([
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [np.nan, np.nan],
    ])
    target = np.column_stack([np.arange(5, dtype=float), np.arange(5, dtype=float)])
    train_mask = np.array([True, True, True, False, False])
    val_mask = np.array([False, False, False, True, False])
    rows = MODULE._ridge_rows(
        fold=1,
        model_name="test_ridge",
        features=features,
        target=target,
        train_mask=train_mask,
        val_mask=val_mask,
        alphas=(1.0,),
        seed=0,
    )
    assert len(rows) == 1
    assert rows[0]["val_samples"] == 1
    assert np.isfinite(rows[0]["val_qlike"])
    assert np.isfinite(rows[0]["val_rmse"])
