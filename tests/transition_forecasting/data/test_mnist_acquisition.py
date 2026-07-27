import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

import transition_forecasting.data.mnist_acquisition as mnist
from transition_forecasting.data.mnist_acquisition import (
    EXPECTED_COUNTS,
    IDX_IMAGE_MAGIC,
    IDX_LABEL_MAGIC,
    _load_mnist_npz,
    discover_idx_files,
    discover_mnist_npz,
)


def _write_idx_header(path: Path, magic: int, count: int, suffix: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(">II", magic, count) + suffix)


def _write_canonical_streams(root: Path) -> dict[str, Path]:
    paths = {
        "train_images": root / "train-images.idx3-ubyte",
        "train_labels": root / "train-labels.idx1-ubyte",
        "test_images": root / "t10k-images.idx3-ubyte",
        "test_labels": root / "t10k-labels.idx1-ubyte",
    }
    _write_idx_header(paths["train_images"], IDX_IMAGE_MAGIC, EXPECTED_COUNTS["train"])
    _write_idx_header(paths["train_labels"], IDX_LABEL_MAGIC, EXPECTED_COUNTS["train"])
    _write_idx_header(paths["test_images"], IDX_IMAGE_MAGIC, EXPECTED_COUNTS["test"])
    _write_idx_header(paths["test_labels"], IDX_LABEL_MAGIC, EXPECTED_COUNTS["test"])
    return paths


def test_discover_idx_files_accepts_identical_duplicate_streams(tmp_path: Path) -> None:
    canonical = _write_canonical_streams(tmp_path)
    duplicate = tmp_path / "nested" / canonical["train_images"].name
    duplicate.parent.mkdir()
    duplicate.write_bytes(canonical["train_images"].read_bytes())

    resolved = discover_idx_files(tmp_path)

    assert resolved == canonical


def test_discover_idx_files_rejects_conflicting_duplicate_streams(tmp_path: Path) -> None:
    canonical = _write_canonical_streams(tmp_path)
    duplicate = tmp_path / "nested" / canonical["train_images"].name
    _write_idx_header(
        duplicate,
        IDX_IMAGE_MAGIC,
        EXPECTED_COUNTS["train"],
        suffix=b"conflict",
    )

    with pytest.raises(ValueError, match="conflicting IDX streams"):
        discover_idx_files(tmp_path)


def test_load_mnist_npz_maps_canonical_array_names(tmp_path: Path) -> None:
    path = tmp_path / "mnist.npz"
    np.savez_compressed(
        path,
        x_train=np.zeros((2, 28, 28), dtype=np.uint8),
        y_train=np.asarray([0, 1], dtype=np.uint8),
        x_test=np.ones((1, 28, 28), dtype=np.uint8),
        y_test=np.asarray([2], dtype=np.uint8),
    )
    arrays = _load_mnist_npz(path)
    assert set(arrays) == {
        "train_images",
        "train_labels",
        "test_images",
        "test_labels",
    }
    assert arrays["train_images"].shape == (2, 28, 28)
    assert arrays["test_labels"].tolist() == [2]


def test_discover_mnist_npz_enforces_pinned_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mnist.npz"
    path.write_bytes(b"checksum-test")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(mnist, "MNIST_NPZ_SHA256", digest)
    assert discover_mnist_npz(tmp_path) == path

    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        discover_mnist_npz(tmp_path)
