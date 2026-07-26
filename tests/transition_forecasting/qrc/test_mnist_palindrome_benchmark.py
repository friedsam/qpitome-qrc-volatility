from __future__ import annotations

import numpy as np
import pytest

from transition_forecasting.qrc.mnist_palindrome_benchmark import (
    MnistPalindromeBenchmarkConfig,
    _stratified_indices,
    average_pool_images,
    images_to_two_channel_sequences,
    local_spatial_contrast,
    shard_global_indices,
)


def test_average_pool_preserves_constant_intensity() -> None:
    images = np.full((2, 28, 28), 128, dtype=np.uint8)
    pooled = average_pool_images(images, output_rows=4, output_cols=4)
    assert pooled.shape == (2, 4, 4)
    np.testing.assert_allclose(pooled, 128.0 / 255.0)


def test_local_contrast_is_zero_for_constant_grid() -> None:
    pooled = np.full((3, 4, 4), 0.25)
    np.testing.assert_allclose(local_spatial_contrast(pooled), 0.0)


def test_two_channel_sequence_has_frozen_length() -> None:
    images = np.zeros((5, 28, 28), dtype=np.uint8)
    images[:, 7:14, 7:14] = 255
    sequence = images_to_two_channel_sequences(
        images,
        output_rows=4,
        output_cols=4,
    )
    assert sequence.shape == (5, 16, 2)
    assert np.isfinite(sequence).all()


def test_stratified_indices_are_balanced_and_deterministic() -> None:
    labels = np.repeat(np.arange(10), 30)
    first = _stratified_indices(labels, 100, 17)
    second = _stratified_indices(labels, 100, 17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        np.bincount(labels[first], minlength=10),
        np.full(10, 10),
    )


def test_shards_cover_each_row_once() -> None:
    shards = [shard_global_indices(23, index, 5) for index in range(5)]
    combined = np.concatenate(shards)
    np.testing.assert_array_equal(np.sort(combined), np.arange(23))
    assert len(np.unique(combined)) == 23


def test_config_rejects_architecture_drift() -> None:
    MnistPalindromeBenchmarkConfig().validate()
    with pytest.raises(ValueError, match="divisible by ten"):
        MnistPalindromeBenchmarkConfig(train_size=201).validate()
    with pytest.raises(ValueError, match="divide the 28x28"):
        MnistPalindromeBenchmarkConfig(pool_rows=5).validate()
    with pytest.raises(ValueError, match="occupation_pair_raw"):
        MnistPalindromeBenchmarkConfig(
            feature_bank="six_mode_density_curvature"
        ).validate()
