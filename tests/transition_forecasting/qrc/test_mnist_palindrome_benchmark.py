from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import transition_forecasting.qrc.mnist_palindrome_benchmark as benchmark
from transition_forecasting.qrc.mnist_palindrome_benchmark import (
    MnistPalindromeBenchmarkConfig,
    _stratified_indices,
    average_pool_images,
    images_to_two_channel_sequences,
    local_spatial_contrast,
    shard_global_indices,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
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


def test_sharded_smoke_merge_writes_complete_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(19)
    arrays = {
        "train_images": rng.integers(0, 256, (600, 28, 28), dtype=np.uint8),
        "test_images": rng.integers(0, 256, (200, 28, 28), dtype=np.uint8),
        "train_labels": np.repeat(np.arange(10), 60),
        "test_labels": np.repeat(np.arange(10), 20),
    }
    monkeypatch.setattr(benchmark, "load_mnist", lambda _: arrays)

    def fake_feature_batch(
        sequences: np.ndarray,
        *_: object,
        interactions: bool,
        **__: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        flat = np.asarray(sequences, dtype=float).reshape(len(sequences), -1)
        phase = np.linspace(0.1, 2.7, 63)[None, :]
        scale = 1.0 if interactions else 0.5
        features = np.sin(scale * flat.sum(axis=1, keepdims=True) + phase)
        return features, {
            "probe_steps": [4, 8, 16],
            "interactions_enabled": interactions,
        }

    monkeypatch.setattr(benchmark, "_feature_batch", fake_feature_batch)
    config = MnistPalindromeBenchmarkConfig(
        train_size=100,
        test_size=20,
        batch_size=7,
        logistic_cs=(0.1, 1.0),
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        omega_mod_fraction=0.60,
        step_duration_us=0.02,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
    )
    geometry = StaggeredLadderGeometryConfig()
    results_root = tmp_path / "results"
    shard_dirs = []
    for shard_index in range(2):
        run_id = f"mnist_smoke_shard_{shard_index:03d}_of_002"
        shard_dir = benchmark.run_mnist_palindrome_feature_shard(
            raw_dir=tmp_path / "raw",
            results_root=results_root,
            run_id=run_id,
            shard_index=shard_index,
            shard_count=2,
            config=config,
            reservoir=reservoir,
            geometry=geometry,
        )
        shard_dirs.append(shard_dir)
        assert benchmark.run_mnist_palindrome_feature_shard(
            raw_dir=tmp_path / "raw",
            results_root=results_root,
            run_id=run_id,
            shard_index=shard_index,
            shard_count=2,
            config=config,
            reservoir=reservoir,
            geometry=geometry,
            resume=True,
        ) == shard_dir

    merged = benchmark.merge_mnist_palindrome_shards(
        shard_dirs=shard_dirs,
        results_root=results_root,
        run_id="mnist_smoke_merged",
        config=config,
    )
    required = {
        "params.json",
        "mnist_palindrome_features.npz",
        "model_comparison.csv",
        "mnist_predictions.csv",
        "per_class_metrics.csv",
        "shard_manifest.csv",
        "summary.json",
    }
    assert required.issubset({path.name for path in merged.iterdir()})
    summary = json.loads((merged / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "mnist_palindrome_benchmark_complete"
    assert summary["train_samples"] == 100
    assert summary["test_samples"] == 20
    assert summary["reservoir"]["qrc_feature_width"] == 63
    assert summary["runtime_seconds"]["shards"] == 2
