"""Focused rolling-fold loader used by the canonical Case151 assay."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.fold_selection import validate_fold_manifest_schema


@dataclass(frozen=True)
class RollingFoldDataset:
    manifest: pd.DataFrame
    values: np.ndarray
    sample_ids: np.ndarray
    folds: np.ndarray
    fold_splits: np.ndarray
    valid: np.ndarray
    channel_names: tuple[str, ...]


def load_rolling_fold_dataset(fold_dir: Path) -> RollingFoldDataset:
    fold_dir = Path(fold_dir)
    manifest_path = fold_dir / "rematched_rolling_manifest.csv"
    tensor_path = fold_dir / "rematched_rolling_tensors.npz"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing fold manifest: {manifest_path}")
    if not tensor_path.is_file():
        raise FileNotFoundError(f"missing fold tensor archive: {tensor_path}")
    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    validate_fold_manifest_schema(manifest)
    with np.load(tensor_path, allow_pickle=False) as bundle:
        required = {"X", "sample_id", "fold", "fold_split"}
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"{tensor_path} missing arrays: {sorted(missing)}")
        values = np.asarray(bundle["X"], dtype=float)
        sample_ids = np.asarray(bundle["sample_id"]).astype(str)
        folds = np.asarray(bundle["fold"], dtype=int)
        fold_splits = np.asarray(bundle["fold_split"]).astype(str)
        valid = (
            np.asarray(bundle["valid"], dtype=bool)
            if "valid" in bundle.files
            else np.ones(len(values), dtype=bool)
        )
        channel_names = (
            tuple(str(value) for value in np.asarray(bundle["channel_names"]))
            if "channel_names" in bundle.files
            else tuple()
        )
    if values.ndim not in (2, 3):
        raise ValueError(f"{tensor_path}: X must be two- or three-dimensional, got {values.shape}")
    expected_length = len(manifest)
    arrays = {
        "X": len(values),
        "sample_id": len(sample_ids),
        "fold": len(folds),
        "fold_split": len(fold_splits),
        "valid": len(valid),
    }
    if any(length != expected_length for length in arrays.values()):
        raise ValueError(
            "fold manifest/tensor lengths are inconsistent: "
            f"manifest={expected_length}, arrays={arrays}"
        )
    if not np.array_equal(sample_ids, manifest["sample_id"].astype(str).to_numpy()):
        raise ValueError("fold manifest and tensor sample IDs are not aligned")
    if not np.array_equal(folds, manifest["fold"].astype(int).to_numpy()):
        raise ValueError("fold manifest and tensor fold values are not aligned")
    if not np.array_equal(
        fold_splits, manifest["fold_split"].astype(str).to_numpy()
    ):
        raise ValueError("fold manifest and tensor split values are not aligned")
    indexed_manifest = manifest.copy()
    indexed_manifest["_tensor_row"] = np.arange(expected_length, dtype=int)
    return RollingFoldDataset(
        manifest=indexed_manifest,
        values=values,
        sample_ids=sample_ids,
        folds=folds,
        fold_splits=fold_splits,
        valid=valid,
        channel_names=channel_names,
    )


def resolve_level_channel(
    dataset: RollingFoldDataset,
    *,
    name: str,
    fallback: int,
) -> int | None:
    if dataset.values.ndim == 2:
        return None
    if name in dataset.channel_names:
        return dataset.channel_names.index(name)
    if fallback < 0 or fallback >= dataset.values.shape[2]:
        raise ValueError(
            f"fallback level channel {fallback} outside tensor shape {dataset.values.shape}"
        )
    return int(fallback)


def extract_level_windows(
    dataset: RollingFoldDataset,
    row_indices: np.ndarray,
    *,
    sequence_length: int,
    level_channel: int | None,
) -> np.ndarray:
    if dataset.values.shape[1] < sequence_length:
        raise ValueError(
            f"requested {sequence_length} steps but tensor has {dataset.values.shape[1]}"
        )
    selected = dataset.values[row_indices, -sequence_length:]
    if selected.ndim == 3:
        if level_channel is None:
            raise ValueError("a channel index is required for a three-dimensional tensor")
        selected = selected[:, :, level_channel]
    level = np.asarray(selected, dtype=float)
    if level.ndim != 2:
        raise ValueError(f"level extraction produced unexpected shape {level.shape}")
    return level
