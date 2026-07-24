import struct
from pathlib import Path

import pytest

from transition_forecasting.data.mnist_acquisition import (
    EXPECTED_COUNTS,
    IDX_IMAGE_MAGIC,
    IDX_LABEL_MAGIC,
    discover_idx_files,
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
