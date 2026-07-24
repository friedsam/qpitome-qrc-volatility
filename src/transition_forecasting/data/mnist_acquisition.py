"""MNIST acquisition with live Kaggle and hash-verified fallback modes."""
from __future__ import annotations

import gzip
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from transition_forecasting.data.acquisition import (
    file_inventory,
    install_candidate,
    utc_now,
    verify_fallback_manifest,
)

DATASET = "hojjatk/mnist-dataset"
DATASET_URL = "https://www.kaggle.com/datasets/hojjatk/mnist-dataset"
LICENSE = "Original MNIST dataset terms; see source manifest"
SOURCE_DESCRIPTION = (
    "MNIST handwritten digit database (LeCun, Cortes, Burges), "
    "original IDX byte streams."
)
IDX_IMAGE_MAGIC = 2051
IDX_LABEL_MAGIC = 2049
EXPECTED_COUNTS = {"train": 60000, "test": 10000}
EXPECTED_IMAGE_SHAPE = (28, 28)


def _open_maybe_gzip(path: Path):
    with path.open("rb") as probe:
        signature = probe.read(2)
    return gzip.open(path, "rb") if signature == b"\x1f\x8b" else path.open("rb")


def read_idx(path: Path) -> np.ndarray:
    """Read a one-to-three-dimensional unsigned-byte IDX stream."""
    with _open_maybe_gzip(path) as handle:
        header = handle.read(4)
        if len(header) != 4:
            raise ValueError(f"{path}: truncated IDX header")
        zero_a, zero_b, dtype_code, n_dims = struct.unpack(">BBBB", header)
        if zero_a != 0 or zero_b != 0:
            raise ValueError(f"{path}: not an IDX stream")
        if dtype_code != 0x08:
            raise ValueError(f"{path}: unsupported IDX dtype code {dtype_code:#x}")
        if not 1 <= n_dims <= 3:
            raise ValueError(f"{path}: unexpected IDX dimension count {n_dims}")
        raw_dims = handle.read(4 * n_dims)
        if len(raw_dims) != 4 * n_dims:
            raise ValueError(f"{path}: truncated IDX dimensions")
        dims = struct.unpack(f">{n_dims}I", raw_dims)
        payload = handle.read()
    expected = int(np.prod(dims))
    if len(payload) != expected:
        raise ValueError(
            f"{path}: payload size {len(payload)} does not match dimensions {dims}"
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(dims)


def _idx_magic(path: Path) -> tuple[int, int] | None:
    try:
        with _open_maybe_gzip(path) as handle:
            header = handle.read(8)
    except OSError:
        return None
    if len(header) != 8:
        return None
    magic, count = struct.unpack(">II", header)
    if magic not in {IDX_IMAGE_MAGIC, IDX_LABEL_MAGIC}:
        return None
    return magic, count


def discover_idx_files(root: Path) -> dict[str, Path]:
    """Locate the four canonical streams by content, not filename."""
    found: dict[str, list[Path]] = {}
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        probed = _idx_magic(path)
        if probed is None:
            continue
        magic, count = probed
        split = next(
            (name for name, expected in EXPECTED_COUNTS.items() if expected == count),
            None,
        )
        if split is None:
            continue
        kind = "images" if magic == IDX_IMAGE_MAGIC else "labels"
        found.setdefault(f"{split}_{kind}", []).append(path)
    required = {"train_images", "train_labels", "test_images", "test_labels"}
    missing = sorted(required.difference(found))
    if missing:
        raise ValueError(f"{root}: could not locate IDX streams for {missing}")
    ambiguous = {
        role: [str(path) for path in paths]
        for role, paths in found.items()
        if len(paths) != 1
    }
    if ambiguous:
        raise ValueError(f"{root}: ambiguous IDX streams: {ambiguous}")
    return {role: paths[0] for role, paths in found.items()}


def validate_source(root: Path) -> dict[str, object]:
    roles = discover_idx_files(root)
    summary: dict[str, object] = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        images = read_idx(roles[f"{split}_images"])
        labels = read_idx(roles[f"{split}_labels"])
        expected_shape = (expected_count, *EXPECTED_IMAGE_SHAPE)
        if images.shape != expected_shape:
            raise ValueError(
                f"{split} images have shape {images.shape}; expected {expected_shape}"
            )
        if labels.shape != (expected_count,):
            raise ValueError(
                f"{split} labels have shape {labels.shape}; expected {(expected_count,)}"
            )
        present = np.unique(labels)
        if present.tolist() != list(range(10)):
            raise ValueError(f"{split} labels do not cover digits 0-9")
        summary[f"{split}_images"] = roles[f"{split}_images"].relative_to(root).as_posix()
        summary[f"{split}_labels"] = roles[f"{split}_labels"].relative_to(root).as_posix()
        summary[f"{split}_count"] = expected_count
    records = file_inventory(root)
    if not records:
        raise ValueError(f"no source files found under {root}")
    summary["file_count"] = len(records)
    summary["files"] = records
    return summary


def write_live_source_manifest(destination: Path) -> None:
    payload = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "source_description": SOURCE_DESCRIPTION,
        "downloaded_at_utc": utc_now(),
        "redistribution_note": (
            "Preserve this manifest and the original MNIST attribution with any copy."
        ),
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def download_live(destination: Path) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("Kaggle CLI not found")
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            kaggle,
            "datasets",
            "download",
            "--dataset",
            DATASET,
            "--path",
            str(destination),
            "--unzip",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    write_live_source_manifest(destination)


def _copy_source(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".DS_Store", "fallback_manifest.json", "raw_acquisition_manifest.json"
        ),
    )


def acquire_mnist(
    destination: Path,
    fallback: Path,
    *,
    source_mode: str = "auto",
    force: bool = False,
) -> dict[str, object]:
    """Resolve MNIST live first, then use the verified fallback snapshot."""
    if source_mode not in {"auto", "live", "fallback"}:
        raise ValueError(f"unsupported source mode: {source_mode}")
    destination = Path(destination)
    fallback = Path(fallback)
    started = utc_now()
    live_error: str | None = None
    selected_mode: str | None = None
    fallback_verification: dict[str, object] | None = None

    if destination.exists() and any(destination.iterdir()) and not force:
        validation = validate_source(destination)
        selected_mode = "existing"
    else:
        with tempfile.TemporaryDirectory(prefix="mnist-raw-") as temporary:
            candidate = Path(temporary) / "candidate"
            if source_mode in {"auto", "live"}:
                try:
                    download_live(candidate)
                    validation = validate_source(candidate)
                    selected_mode = "live"
                except Exception as exc:
                    live_error = f"{type(exc).__name__}: {exc}"
                    if candidate.exists():
                        shutil.rmtree(candidate)
                    if source_mode == "live":
                        raise
            if selected_mode is None:
                fallback_verification = verify_fallback_manifest(fallback)
                validate_source(fallback)
                _copy_source(fallback, candidate)
                validation = validate_source(candidate)
                selected_mode = "fallback"
            install_candidate(candidate, destination, force=force)

    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "source_description": SOURCE_DESCRIPTION,
        "source_mode_requested": source_mode,
        "source_mode_used": selected_mode,
        "live_retrieval_error": live_error,
        "fallback_path": str(fallback),
        "fallback_verification": fallback_verification,
        "destination": str(destination),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "validation": validation,
    }
    (destination / "raw_acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_mnist(root: Path) -> dict[str, np.ndarray]:
    roles = discover_idx_files(root)
    return {role: read_idx(path) for role, path in roles.items()}


def write_fallback_manifest(root: Path) -> Path:
    records = file_inventory(root)
    if not records:
        raise ValueError(f"no files to record under {root}")
    payload = {
        "schema_version": 1,
        "dataset": DATASET,
        "role": "repository fallback source snapshot",
        "source_boundary": "raw MNIST IDX input",
        "generated_at_utc": utc_now(),
        "file_count": len(records),
        "files": records,
    }
    path = Path(root) / "fallback_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
