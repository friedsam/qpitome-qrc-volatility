"""Submission-oriented MNIST benchmark for the frozen A/B/A palindrome QRC.

The primary benchmark deliberately reuses the financial model's six-atom
staggered ladder, palindromic A/4-B/2-A/4 control schedule, hardware-natural
occupation/pair observables, and train-only channel scaling. Only the task
adapter and multinomial classical readout differ.

Expensive feature generation is resumable and shardable. Each shard records a
configuration fingerprint and global sample indices; merging rejects missing,
duplicated, or incompatible rows.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from transition_forecasting.data.mnist_acquisition import load_mnist
from transition_forecasting.qrc.bivariate_crossover_assay import (
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    _resolve_schedule,
    evolve_palindrome_probabilities,
)
from transition_forecasting.qrc.representation_candidates import (
    ChannelScaler,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


@dataclass(frozen=True)
class MnistPalindromeBenchmarkConfig:
    """Frozen protocol for the primary common MNIST benchmark."""

    train_size: int = 2000
    test_size: int = 1000
    subset_seed: int = 20260726
    pool_rows: int = 4
    pool_cols: int = 4
    q_low: float = 0.01
    q_high: float = 0.99
    logistic_cs: tuple[float, ...] = (0.1, 1.0, 10.0)
    inner_validation_fraction: float = 0.20
    readout_seed: int = 20260726
    batch_size: int = 32
    schedule_name: str = "crossover_Ahalf_B_Ahalf"
    feature_bank: str = "occupation_pair_raw"
    include_interaction_off: bool = False

    def validate(self) -> None:
        if self.train_size < 100 or self.test_size < 20:
            raise ValueError("train_size/test_size are too small for ten-class MNIST")
        if self.train_size % 10 or self.test_size % 10:
            raise ValueError("train_size and test_size must be divisible by ten")
        if self.pool_rows < 1 or self.pool_cols < 1:
            raise ValueError("pool dimensions must be positive")
        if 28 % self.pool_rows or 28 % self.pool_cols:
            raise ValueError("pool dimensions must divide the 28x28 MNIST image")
        if not 0.0 <= self.q_low < self.q_high <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= q_low < q_high <= 1")
        if not self.logistic_cs or any(value <= 0 for value in self.logistic_cs):
            raise ValueError("logistic_cs must be positive and nonempty")
        if len(set(self.logistic_cs)) != len(self.logistic_cs):
            raise ValueError("logistic_cs must be unique")
        if not 0.1 <= self.inner_validation_fraction <= 0.4:
            raise ValueError("inner_validation_fraction must lie in [0.1, 0.4]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.feature_bank != "occupation_pair_raw":
            raise ValueError("primary MNIST benchmark must use occupation_pair_raw")
        _resolve_schedule(self.schedule_name)

    @property
    def sequence_length(self) -> int:
        return int(self.pool_rows * self.pool_cols)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["sequence_length"] = self.sequence_length
        return payload


@dataclass(frozen=True)
class PreparedMnistPalindrome:
    sample_id: np.ndarray
    split: np.ndarray
    source_index: np.ndarray
    labels: np.ndarray
    raw_sequences: np.ndarray
    encoded_sequences: np.ndarray
    input_features: np.ndarray
    train_mask: np.ndarray
    scaler: ChannelScaler
    dataset_fingerprint: str


@dataclass(frozen=True)
class ReadoutResult:
    model_name: str
    selected_c: float | None
    inner_accuracy: float | None
    inner_macro_f1: float | None
    test_accuracy: float
    test_macro_f1: float
    predictions: np.ndarray
    training_seconds: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _update_array_hash(digest: "hashlib._Hash", array: np.ndarray) -> None:
    values = np.ascontiguousarray(array)
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


def dataset_fingerprint(
    sample_id: np.ndarray,
    split: np.ndarray,
    source_index: np.ndarray,
    labels: np.ndarray,
    raw_sequences: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for value in (sample_id.astype("U"), split.astype("U")):
        digest.update("\n".join(value.tolist()).encode("utf-8"))
        digest.update(b"\0")
    for value in (source_index, labels, raw_sequences):
        _update_array_hash(digest, value)
    return digest.hexdigest()


def _stratified_indices(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    target = np.asarray(labels, dtype=int).reshape(-1)
    if size % 10:
        raise ValueError("stratified MNIST subset size must be divisible by ten")
    per_class = size // 10
    rng = np.random.default_rng(int(seed))
    blocks: list[np.ndarray] = []
    for label in range(10):
        available = np.flatnonzero(target == label)
        if len(available) < per_class:
            raise ValueError(f"digit {label} has only {len(available)} rows")
        blocks.append(rng.choice(available, size=per_class, replace=False))
    selected = np.concatenate(blocks)
    rng.shuffle(selected)
    return selected.astype(np.int64)


def average_pool_images(
    images: np.ndarray,
    *,
    output_rows: int = 4,
    output_cols: int = 4,
) -> np.ndarray:
    """Average-pool uint8 or floating images into one deterministic coarse grid."""

    values = np.asarray(images, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (28, 28):
        raise ValueError("images must have shape (samples, 28, 28)")
    if 28 % output_rows or 28 % output_cols:
        raise ValueError("output dimensions must divide 28")
    row_width = 28 // output_rows
    col_width = 28 // output_cols
    pooled = values.reshape(
        len(values), output_rows, row_width, output_cols, col_width
    ).mean(axis=(2, 4))
    return pooled / 255.0 if values.max(initial=0.0) > 1.0 else pooled


def local_spatial_contrast(pooled: np.ndarray) -> np.ndarray:
    """Return center minus mean orthogonal-neighbour intensity at every cell."""

    values = np.asarray(pooled, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("pooled images must be finite with shape (samples, rows, cols)")
    neighbour_sum = np.zeros_like(values)
    neighbour_count = np.zeros(values.shape[1:], dtype=float)

    neighbour_sum[:, 1:, :] += values[:, :-1, :]
    neighbour_count[1:, :] += 1.0
    neighbour_sum[:, :-1, :] += values[:, 1:, :]
    neighbour_count[:-1, :] += 1.0
    neighbour_sum[:, :, 1:] += values[:, :, :-1]
    neighbour_count[:, 1:] += 1.0
    neighbour_sum[:, :, :-1] += values[:, :, 1:]
    neighbour_count[:, :-1] += 1.0

    if np.any(neighbour_count <= 0):
        raise RuntimeError("spatial contrast encountered an isolated grid cell")
    return values - neighbour_sum / neighbour_count[None, :, :]


def images_to_two_channel_sequences(
    images: np.ndarray,
    *,
    output_rows: int,
    output_cols: int,
) -> np.ndarray:
    pooled = average_pool_images(
        images,
        output_rows=output_rows,
        output_cols=output_cols,
    )
    contrast = local_spatial_contrast(pooled)
    intensity_sequence = pooled.reshape(len(pooled), -1)
    contrast_sequence = contrast.reshape(len(contrast), -1)
    return np.stack([intensity_sequence, contrast_sequence], axis=-1)


def prepare_fixed_mnist(
    arrays: dict[str, np.ndarray],
    config: MnistPalindromeBenchmarkConfig,
) -> PreparedMnistPalindrome:
    """Create the deterministic official-split subset and train-only scaling."""

    config.validate()
    train_images = np.asarray(arrays["train_images"], dtype=np.uint8)
    test_images = np.asarray(arrays["test_images"], dtype=np.uint8)
    train_labels = np.asarray(arrays["train_labels"], dtype=int).reshape(-1)
    test_labels = np.asarray(arrays["test_labels"], dtype=int).reshape(-1)

    train_index = _stratified_indices(
        train_labels,
        config.train_size,
        config.subset_seed,
    )
    test_index = _stratified_indices(
        test_labels,
        config.test_size,
        config.subset_seed + 1,
    )
    train_sequence = images_to_two_channel_sequences(
        train_images[train_index],
        output_rows=config.pool_rows,
        output_cols=config.pool_cols,
    )
    test_sequence = images_to_two_channel_sequences(
        test_images[test_index],
        output_rows=config.pool_rows,
        output_cols=config.pool_cols,
    )
    raw = np.concatenate([train_sequence, test_sequence], axis=0)
    labels = np.concatenate([train_labels[train_index], test_labels[test_index]])
    split = np.concatenate(
        [
            np.repeat("train", len(train_index)),
            np.repeat("test", len(test_index)),
        ]
    ).astype("U5")
    source_index = np.concatenate([train_index, test_index]).astype(np.int64)
    sample_id = np.concatenate(
        [
            np.asarray([f"train_{value:05d}" for value in train_index]),
            np.asarray([f"test_{value:05d}" for value in test_index]),
        ]
    ).astype("U16")
    train_mask = split == "train"
    scaler = fit_channel_scaler(
        raw,
        train_mask,
        q_low=config.q_low,
        q_high=config.q_high,
    )
    encoded = transform_candidate_sequences(raw, scaler)
    inputs = encoded.reshape(len(encoded), -1)
    fingerprint = dataset_fingerprint(
        sample_id,
        split,
        source_index,
        labels,
        raw,
    )
    return PreparedMnistPalindrome(
        sample_id=sample_id,
        split=split,
        source_index=source_index,
        labels=labels.astype(np.int64),
        raw_sequences=raw,
        encoded_sequences=encoded,
        input_features=inputs,
        train_mask=train_mask,
        scaler=scaler,
        dataset_fingerprint=fingerprint,
    )


def shard_global_indices(total_rows: int, shard_index: int, shard_count: int) -> np.ndarray:
    if total_rows < 1:
        raise ValueError("total_rows must be positive")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    return np.arange(shard_index, total_rows, shard_count, dtype=np.int64)


def benchmark_identity(
    config: MnistPalindromeBenchmarkConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float,
    drive_phase_rad: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": "mnist_palindrome_qrc",
        "config": config.to_dict(),
        "reservoir": reservoir.to_dict(),
        "geometry": geometry.to_dict(),
        "interaction_scale": float(interaction_scale),
        "drive_phase_rad": float(drive_phase_rad),
        "schedule": config.schedule_name,
        "feature_bank": config.feature_bank,
    }


def _validate_primary_reservoir(
    config: MnistPalindromeBenchmarkConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float,
    drive_phase_rad: float,
) -> None:
    config.validate()
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("primary MNIST benchmark requires an exact six-atom reservoir")
    if not np.isclose(reservoir.step_duration_us, 0.02):
        raise ValueError("primary MNIST benchmark requires 0.02-us palindrome steps")
    if tuple(reservoir.probe_fractions) != (0.25, 0.5, 1.0):
        raise ValueError("primary MNIST benchmark requires probes at 1/4, 1/2, and endpoint")
    if interaction_scale <= 0:
        raise ValueError("interaction_scale must be positive")
    if not np.isfinite(drive_phase_rad):
        raise ValueError("drive_phase_rad must be finite")


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_batch(
    sequences: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    schedule_name: str,
    interaction_scale: float,
    drive_phase_rad: float,
    interactions: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    schedule = _resolve_schedule(schedule_name)
    probabilities, metadata = evolve_palindrome_probabilities(
        sequences,
        reservoir,
        geometry,
        schedule,
        interaction_scale=interaction_scale,
        interactions=interactions,
        drive_phase_rad=drive_phase_rad,
    )
    matrix = np.asarray(
        build_crossover_feature_banks(probabilities)["occupation_pair_raw"],
        dtype=float,
    )
    if matrix.ndim != 2 or len(matrix) != len(sequences) or not np.isfinite(matrix).all():
        raise RuntimeError("palindrome feature batch is malformed")
    return matrix, metadata


def _batch_paths(batch_root: Path, batch_number: int) -> tuple[Path, Path]:
    stem = f"batch_{batch_number:05d}"
    return batch_root / f"{stem}.npz", batch_root / f"{stem}.json"


def _validate_existing_batch(
    npz_path: Path,
    metadata_path: Path,
    *,
    expected_indices: np.ndarray,
    identity_sha256: str,
    dataset_sha256: str,
) -> None:
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete cached batch: {npz_path}")
    metadata = _read_json(metadata_path)
    if metadata.get("identity_sha256") != identity_sha256:
        raise ValueError(f"cached batch identity mismatch: {metadata_path}")
    if metadata.get("dataset_sha256") != dataset_sha256:
        raise ValueError(f"cached batch dataset mismatch: {metadata_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        observed = np.asarray(payload["global_index"], dtype=np.int64)
        features = np.asarray(payload["qrc_features"], dtype=float)
    if not np.array_equal(observed, expected_indices):
        raise ValueError(f"cached batch row mismatch: {npz_path}")
    if features.ndim != 2 or len(features) != len(observed) or not np.isfinite(features).all():
        raise ValueError(f"cached batch features are invalid: {npz_path}")


def run_mnist_palindrome_feature_shard(
    *,
    raw_dir: Path,
    results_root: Path,
    run_id: str,
    shard_index: int,
    shard_count: int,
    config: MnistPalindromeBenchmarkConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float = 1.25,
    drive_phase_rad: float = 0.0,
    resume: bool = False,
) -> Path:
    """Generate one resumable feature shard and its complete provenance package."""

    _validate_primary_reservoir(
        config,
        reservoir,
        geometry,
        interaction_scale=interaction_scale,
        drive_phase_rad=drive_phase_rad,
    )
    arrays = load_mnist(Path(raw_dir))
    prepared = prepare_fixed_mnist(arrays, config)
    global_indices = shard_global_indices(len(prepared.labels), shard_index, shard_count)
    identity = benchmark_identity(
        config,
        reservoir,
        geometry,
        interaction_scale=interaction_scale,
        drive_phase_rad=drive_phase_rad,
    )
    identity_sha256 = payload_sha256(identity)
    run_dir = Path(results_root) / run_id
    params_path = run_dir / "params.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(run_dir)
        if not params_path.is_file():
            raise ValueError(f"existing run lacks params.json: {run_dir}")
        existing = _read_json(params_path)
        if existing.get("identity_sha256") != identity_sha256:
            raise ValueError("resume identity differs from existing shard")
        if existing.get("dataset_sha256") != prepared.dataset_fingerprint:
            raise ValueError("resume dataset differs from existing shard")
        if existing.get("shard_index") != shard_index or existing.get("shard_count") != shard_count:
            raise ValueError("resume shard coordinates differ from existing shard")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            params_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "created_at_utc": utc_now(),
                "raw_dir": str(raw_dir),
                "identity": identity,
                "identity_sha256": identity_sha256,
                "dataset_sha256": prepared.dataset_fingerprint,
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
                "global_rows": int(len(prepared.labels)),
                "shard_rows": int(len(global_indices)),
                "scaler": prepared.scaler.to_dict(),
            },
        )
        _atomic_savez(
            run_dir / "sample_contract.npz",
            sample_id=prepared.sample_id,
            split=prepared.split,
            source_index=prepared.source_index,
            labels=prepared.labels,
            train_mask=prepared.train_mask,
        )

    batch_root = run_dir / "batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_records: list[dict[str, object]] = []
    started_total = time.perf_counter()
    for batch_number, start in enumerate(range(0, len(global_indices), config.batch_size)):
        stop = min(start + config.batch_size, len(global_indices))
        indices = global_indices[start:stop]
        npz_path, metadata_path = _batch_paths(batch_root, batch_number)
        if npz_path.exists() or metadata_path.exists():
            if not resume:
                raise FileExistsError(f"batch output already exists: {npz_path}")
            _validate_existing_batch(
                npz_path,
                metadata_path,
                expected_indices=indices,
                identity_sha256=identity_sha256,
                dataset_sha256=prepared.dataset_fingerprint,
            )
            batch_records.append(_read_json(metadata_path))
            continue

        batch_started = time.perf_counter()
        qrc_on, metadata_on = _feature_batch(
            prepared.encoded_sequences[indices],
            reservoir,
            geometry,
            schedule_name=config.schedule_name,
            interaction_scale=interaction_scale,
            drive_phase_rad=drive_phase_rad,
            interactions=True,
        )
        arrays_to_save: dict[str, np.ndarray] = {
            "global_index": indices,
            "sample_id": prepared.sample_id[indices],
            "split": prepared.split[indices],
            "source_index": prepared.source_index[indices],
            "labels": prepared.labels[indices],
            "input_features": prepared.input_features[indices],
            "qrc_features": qrc_on,
        }
        metadata_off: dict[str, object] | None = None
        if config.include_interaction_off:
            qrc_off, metadata_off = _feature_batch(
                prepared.encoded_sequences[indices],
                reservoir,
                geometry,
                schedule_name=config.schedule_name,
                interaction_scale=interaction_scale,
                drive_phase_rad=drive_phase_rad,
                interactions=False,
            )
            if qrc_off.shape != qrc_on.shape:
                raise RuntimeError("interaction-off feature width differs from interaction-on")
            arrays_to_save["interaction_off_features"] = qrc_off
        _atomic_savez(npz_path, **arrays_to_save)
        record = {
            "schema_version": 1,
            "batch_number": int(batch_number),
            "global_start": int(indices[0]),
            "global_stop_inclusive": int(indices[-1]),
            "rows": int(len(indices)),
            "feature_width": int(qrc_on.shape[1]),
            "identity_sha256": identity_sha256,
            "dataset_sha256": prepared.dataset_fingerprint,
            "wall_seconds": float(time.perf_counter() - batch_started),
            "interactions_on_metadata": metadata_on,
            "interactions_off_metadata": metadata_off,
        }
        _write_json(metadata_path, record)
        batch_records.append(record)

    batch_npz = sorted(batch_root.glob("batch_*.npz"))
    if len(batch_npz) != len(batch_records):
        raise RuntimeError("batch count mismatch after shard generation")
    blocks: list[dict[str, np.ndarray]] = []
    for path in batch_npz:
        with np.load(path, allow_pickle=False) as payload:
            blocks.append({name: np.asarray(payload[name]) for name in payload.files})
    merged_indices = np.concatenate([block["global_index"] for block in blocks])
    order = np.argsort(merged_indices)
    if not np.array_equal(merged_indices[order], np.sort(global_indices)):
        raise RuntimeError("shard merge lost or duplicated global indices")
    output_arrays: dict[str, np.ndarray] = {}
    for name in blocks[0]:
        output_arrays[name] = np.concatenate([block[name] for block in blocks], axis=0)[order]
    _atomic_savez(run_dir / "features.npz", **output_arrays)
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "status": "mnist_palindrome_feature_shard_complete",
            "run_id": run_id,
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "rows": int(len(global_indices)),
            "global_rows": int(len(prepared.labels)),
            "identity_sha256": identity_sha256,
            "dataset_sha256": prepared.dataset_fingerprint,
            "feature_width": int(output_arrays["qrc_features"].shape[1]),
            "include_interaction_off": bool(config.include_interaction_off),
            "wall_seconds": float(time.perf_counter() - started_total),
            "batch_count": int(len(batch_records)),
            "files": {
                "features": "features.npz",
                "sample_contract": "sample_contract.npz",
                "params": "params.json",
            },
        },
    )
    return run_dir


def _load_shard(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    summary = _read_json(path / "summary.json")
    params = _read_json(path / "params.json")
    with np.load(path / "features.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if summary.get("status") != "mnist_palindrome_feature_shard_complete":
        raise ValueError(f"incomplete MNIST shard: {path}")
    return params, arrays


def merge_mnist_palindrome_shards(
    *,
    shard_dirs: Iterable[Path],
    results_root: Path,
    run_id: str,
    config: MnistPalindromeBenchmarkConfig,
) -> Path:
    """Merge complete shard runs, validate coverage, and train final readouts."""

    paths = [Path(value) for value in shard_dirs]
    if not paths:
        raise ValueError("at least one shard directory is required")
    records = [_load_shard(path) for path in paths]
    params = [item[0] for item in records]
    identities = {str(item["identity_sha256"]) for item in params}
    datasets = {str(item["dataset_sha256"]) for item in params}
    shard_counts = {int(item["shard_count"]) for item in params}
    if len(identities) != 1 or len(datasets) != 1 or len(shard_counts) != 1:
        raise ValueError("shards do not share one identity, dataset, and shard count")
    if params[0]["identity"]["config"] != config.to_dict():
        raise ValueError("merge config differs from the shard identity")
    shard_count = shard_counts.pop()
    shard_indices = sorted(int(item["shard_index"]) for item in params)
    if shard_indices != list(range(shard_count)):
        raise ValueError(
            f"shard coverage mismatch: expected {list(range(shard_count))}, observed {shard_indices}"
        )

    arrays = [item[1] for item in records]
    available_fields = set(arrays[0])
    if any(set(item) != available_fields for item in arrays[1:]):
        raise ValueError("shards contain different feature fields")
    merged = {
        name: np.concatenate([item[name] for item in arrays], axis=0)
        for name in sorted(available_fields)
    }
    global_index = np.asarray(merged["global_index"], dtype=np.int64)
    order = np.argsort(global_index)
    expected_rows = int(params[0]["global_rows"])
    if not np.array_equal(global_index[order], np.arange(expected_rows, dtype=np.int64)):
        raise ValueError("merged shards contain missing or duplicate global rows")
    merged = {name: value[order] for name, value in merged.items()}
    if len(np.unique(merged["sample_id"])) != expected_rows:
        raise ValueError("merged sample identifiers are not unique")
    if not np.isfinite(np.asarray(merged["qrc_features"], dtype=float)).all():
        raise ValueError("merged QRC features contain non-finite values")

    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_savez(run_dir / "mnist_palindrome_features.npz", **merged)
    _write_json(
        run_dir / "params.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "identity_sha256": next(iter(identities)),
            "dataset_sha256": next(iter(datasets)),
            "source_shards": [str(path) for path in paths],
            "config": config.to_dict(),
        },
    )
    _finalize_mnist_readouts(run_dir, merged, config)
    return run_dir


def _inner_split(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(labels, dtype=int).reshape(-1)
    rng = np.random.default_rng(int(seed))
    fit: list[int] = []
    tune: list[int] = []
    for label in range(10):
        indices = np.flatnonzero(target == label)
        rng.shuffle(indices)
        tune_count = max(1, int(round(len(indices) * fraction)))
        tune.extend(indices[:tune_count].tolist())
        fit.extend(indices[tune_count:].tolist())
    fit_array = np.asarray(sorted(fit), dtype=np.int64)
    tune_array = np.asarray(sorted(tune), dtype=np.int64)
    if not len(fit_array) or not len(tune_array):
        raise RuntimeError("inner split produced an empty partition")
    return fit_array, tune_array


def _fit_logistic_readout(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    config: MnistPalindromeBenchmarkConfig,
) -> ReadoutResult:
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=int).reshape(-1)
    train = np.asarray(train_mask, dtype=bool)
    if matrix.ndim != 2 or len(matrix) != len(target) or train.shape != (len(target),):
        raise ValueError("readout inputs are not aligned")
    if not train.any() or train.all() or not np.isfinite(matrix).all():
        raise ValueError("readout requires finite train and test features")

    train_rows = np.flatnonzero(train)
    fit_local, tune_local = _inner_split(
        target[train_rows],
        config.inner_validation_fraction,
        config.readout_seed,
    )
    fit_rows = train_rows[fit_local]
    tune_rows = train_rows[tune_local]
    candidates: list[tuple[float, float, float]] = []
    started = time.perf_counter()
    for c_value in config.logistic_cs:
        scaler = StandardScaler().fit(matrix[fit_rows])
        model = LogisticRegression(
            C=float(c_value),
            max_iter=3000,
            solver="lbfgs",
            random_state=config.readout_seed,
        )
        model.fit(scaler.transform(matrix[fit_rows]), target[fit_rows])
        predicted = model.predict(scaler.transform(matrix[tune_rows]))
        macro = float(
            f1_score(target[tune_rows], predicted, average="macro", zero_division=0)
        )
        accuracy = float(accuracy_score(target[tune_rows], predicted))
        candidates.append((macro, accuracy, float(c_value)))
    selected_macro, selected_accuracy, selected_c = sorted(
        candidates,
        key=lambda item: (-item[0], -item[1], item[2]),
    )[0]
    scaler = StandardScaler().fit(matrix[train])
    model = LogisticRegression(
        C=selected_c,
        max_iter=3000,
        solver="lbfgs",
        random_state=config.readout_seed,
    )
    model.fit(scaler.transform(matrix[train]), target[train])
    predictions = model.predict(scaler.transform(matrix[~train])).astype(np.int64)
    elapsed = float(time.perf_counter() - started)
    return ReadoutResult(
        model_name=model_name,
        selected_c=selected_c,
        inner_accuracy=selected_accuracy,
        inner_macro_f1=selected_macro,
        test_accuracy=float(accuracy_score(target[~train], predictions)),
        test_macro_f1=float(
            f1_score(target[~train], predictions, average="macro", zero_division=0)
        ),
        predictions=predictions,
        training_seconds=elapsed,
    )


def _majority_result(labels: np.ndarray, train_mask: np.ndarray) -> ReadoutResult:
    target = np.asarray(labels, dtype=int)
    train = np.asarray(train_mask, dtype=bool)
    majority = int(np.argmax(np.bincount(target[train], minlength=10)))
    predictions = np.repeat(majority, int((~train).sum())).astype(np.int64)
    return ReadoutResult(
        model_name="majority_class",
        selected_c=None,
        inner_accuracy=None,
        inner_macro_f1=None,
        test_accuracy=float(accuracy_score(target[~train], predictions)),
        test_macro_f1=float(
            f1_score(target[~train], predictions, average="macro", zero_division=0)
        ),
        predictions=predictions,
        training_seconds=0.0,
    )


def _finalize_mnist_readouts(
    run_dir: Path,
    merged: dict[str, np.ndarray],
    config: MnistPalindromeBenchmarkConfig,
) -> None:
    labels = np.asarray(merged["labels"], dtype=int)
    split = np.asarray(merged["split"]).astype("U")
    train_mask = split == "train"
    test_mask = split == "test"
    if int(train_mask.sum()) != config.train_size or int(test_mask.sum()) != config.test_size:
        raise ValueError("merged split sizes differ from the frozen config")
    models: list[ReadoutResult] = [_majority_result(labels, train_mask)]
    models.append(
        _fit_logistic_readout(
            "two_channel_input_logistic",
            np.asarray(merged["input_features"], dtype=float),
            labels,
            train_mask,
            config,
        )
    )
    models.append(
        _fit_logistic_readout(
            "palindrome_qrc_logistic",
            np.asarray(merged["qrc_features"], dtype=float),
            labels,
            train_mask,
            config,
        )
    )
    if "interaction_off_features" in merged:
        models.append(
            _fit_logistic_readout(
                "palindrome_interactions_off_logistic",
                np.asarray(merged["interaction_off_features"], dtype=float),
                labels,
                train_mask,
                config,
            )
        )

    comparison_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    per_class_rows: list[dict[str, object]] = []
    confusion_outputs: dict[str, str] = {}
    y_test = labels[test_mask]
    sample_test = np.asarray(merged["sample_id"])[test_mask].astype(str)
    source_test = np.asarray(merged["source_index"])[test_mask].astype(int)
    for result in models:
        comparison_rows.append(
            {
                "model": result.model_name,
                "selected_c": result.selected_c,
                "inner_accuracy": result.inner_accuracy,
                "inner_macro_f1": result.inner_macro_f1,
                "test_accuracy": result.test_accuracy,
                "test_macro_f1": result.test_macro_f1,
                "readout_training_seconds": result.training_seconds,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "model": result.model_name,
                    "sample_id": sample_test,
                    "source_index": source_test,
                    "y_true": y_test,
                    "y_pred": result.predictions,
                    "correct": result.predictions == y_test,
                }
            )
        )
        report = classification_report(
            y_test,
            result.predictions,
            labels=list(range(10)),
            output_dict=True,
            zero_division=0,
        )
        for label in range(10):
            row = report[str(label)]
            per_class_rows.append(
                {
                    "model": result.model_name,
                    "digit": label,
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1-score"]),
                    "support": int(row["support"]),
                }
            )
        matrix = confusion_matrix(y_test, result.predictions, labels=list(range(10)))
        filename = f"confusion_matrix_{result.model_name}.csv"
        pd.DataFrame(
            matrix,
            index=[f"true_{value}" for value in range(10)],
            columns=[f"pred_{value}" for value in range(10)],
        ).to_csv(run_dir / filename)
        confusion_outputs[result.model_name] = filename

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["test_accuracy", "test_macro_f1"], ascending=False
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    per_class = pd.DataFrame(per_class_rows)
    comparison.to_csv(run_dir / "model_comparison.csv", index=False)
    predictions.to_csv(run_dir / "mnist_predictions.csv", index=False)
    per_class.to_csv(run_dir / "per_class_metrics.csv", index=False)

    primary = comparison.loc[comparison["model"].eq("palindrome_qrc_logistic")].iloc[0]
    input_baseline = comparison.loc[
        comparison["model"].eq("two_channel_input_logistic")
    ].iloc[0]
    summary = {
        "schema_version": 1,
        "status": "mnist_palindrome_benchmark_complete",
        "task": "MNIST ten-class classification",
        "official_train_test_partitions_preserved": True,
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "classes": 10,
        "preprocessing": {
            "method": "deterministic average pooling plus local orthogonal-neighbour contrast",
            "pool_shape": [config.pool_rows, config.pool_cols],
            "sequence_length": config.sequence_length,
            "channel_scaling": "train-only robust median/quantile scaling clipped to [-1, 1]",
        },
        "reservoir": {
            "architecture": "six-atom staggered ladder A/4-B/2-A/4 palindrome",
            "feature_bank": config.feature_bank,
            "qrc_feature_width": int(np.asarray(merged["qrc_features"]).shape[1]),
        },
        "primary_metrics": {
            "accuracy": float(primary["test_accuracy"]),
            "macro_f1": float(primary["test_macro_f1"]),
            "accuracy_delta_vs_two_channel_input": float(
                primary["test_accuracy"] - input_baseline["test_accuracy"]
            ),
            "macro_f1_delta_vs_two_channel_input": float(
                primary["test_macro_f1"] - input_baseline["test_macro_f1"]
            ),
        },
        "known_limitations": [
            "A bounded deterministic subset is used to keep exact-state feature generation reproducible.",
            "Row-major pooled-image order is a deterministic task adapter, not a claim that images are naturally temporal.",
            "The classical multinomial readout is trained; the quantum reservoir parameters remain fixed.",
            "A position-encoded reservoir remains a separate optional task-specialized benchmark and is not mixed into this primary result.",
        ],
        "files": {
            "features": "mnist_palindrome_features.npz",
            "model_comparison": "model_comparison.csv",
            "predictions": "mnist_predictions.csv",
            "per_class_metrics": "per_class_metrics.csv",
            "confusion_matrices": confusion_outputs,
        },
    }
    _write_json(run_dir / "summary.json", summary)
