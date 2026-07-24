from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from transition_forecasting.data.mnist_acquisition import read_idx
from transition_forecasting.qrc.mnist_position_benchmark import (
    MnistPositionBenchmarkConfig,
    _stratified_indices,
)
from transition_forecasting.qrc.position_encoded_rydberg import (
    PositionEncodedConfig,
    build_position_encoded_features,
    encode_positions,
)


def _write_idx(path: Path, array: np.ndarray) -> None:
    matrix = np.asarray(array, dtype=np.uint8)
    header = struct.pack(">BBBB", 0, 0, 0x08, matrix.ndim)
    dimensions = struct.pack(f">{matrix.ndim}I", *matrix.shape)
    with gzip.open(path, "wb") as handle:
        handle.write(header + dimensions + matrix.tobytes())


def test_read_idx_round_trip(tmp_path: Path) -> None:
    expected = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    path = tmp_path / "sample.gz"
    _write_idx(path, expected)
    np.testing.assert_array_equal(read_idx(path), expected)


def test_stratified_indices_are_balanced_and_deterministic() -> None:
    labels = np.repeat(np.arange(10), 20)
    first = _stratified_indices(labels, 100, 7)
    second = _stratified_indices(labels, 100, 7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        np.bincount(labels[first], minlength=10), np.full(10, 10)
    )


def test_position_encoding_shape_and_direction() -> None:
    config = PositionEncodedConfig(n_atoms=4)
    positions = encode_positions(np.asarray([[0.0, 0.5, 1.0]]), config)
    gaps = np.diff(positions[0, :, 0])
    assert positions.shape == (1, 4, 2)
    assert gaps[0] > gaps[1] > gaps[2]


def test_interactions_off_erases_sample_identity() -> None:
    config = PositionEncodedConfig(
        n_atoms=4,
        total_time_us=0.08,
        probe_fractions=(1.0,),
        max_phase_per_substep=1.0,
    )
    values = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    features_off, _ = build_position_encoded_features(
        values, config, interactions=False
    )
    features_on, _ = build_position_encoded_features(
        values, config, interactions=True
    )
    np.testing.assert_allclose(features_off[0], features_off[1], atol=1e-12)
    assert not np.allclose(features_on[0], features_on[1])


def test_config_rejects_non_ten_class_sizes() -> None:
    config = MnistPositionBenchmarkConfig(train_size=101, test_size=20)
    try:
        config.validate()
    except ValueError as exc:
        assert "divisible by ten" in str(exc)
    else:
        raise AssertionError("expected invalid size rejection")
