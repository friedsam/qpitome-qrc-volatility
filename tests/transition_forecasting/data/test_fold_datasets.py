from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.data.fold_datasets import (
    DEFAULT_N_FOLDS,
    _load_dataset,
)


def test_default_fold_count_is_eight() -> None:
    assert DEFAULT_N_FOLDS == 8


def test_load_dataset_accepts_only_one_channel(tmp_path: Path) -> None:
    manifest = pd.DataFrame({"sample_id": ["a", "b"]})
    manifest.to_csv(tmp_path / "sample_manifest.csv", index=False)
    np.savez_compressed(
        tmp_path / "sequence_tensors.npz",
        X=np.zeros((2, 40, 1), dtype=float),
        sample_id=np.asarray(["a", "b"], dtype="U1"),
    )

    loaded_manifest, tensor = _load_dataset(tmp_path)

    assert loaded_manifest["sample_id"].tolist() == ["a", "b"]
    assert tensor.shape == (2, 40, 1)


def test_load_dataset_rejects_redundant_three_channel_tensor(tmp_path: Path) -> None:
    manifest = pd.DataFrame({"sample_id": ["a"]})
    manifest.to_csv(tmp_path / "sample_manifest.csv", index=False)
    np.savez_compressed(
        tmp_path / "sequence_tensors.npz",
        X=np.zeros((1, 40, 3), dtype=float),
        sample_id=np.asarray(["a"], dtype="U1"),
    )

    with pytest.raises(ValueError, match="one-channel"):
        _load_dataset(tmp_path)
