from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from transition_forecasting.qrc.residual_head_root_cause_archive import (
    load_ladder_feature_archives,
)


def _write_fold_archive(path: Path, *, fold: int) -> None:
    rows = 6
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fold=np.full(rows, fold, dtype=int),
        sample_id=np.asarray([f"sample_{fold}_{row}" for row in range(rows)]),
        fold_split=np.asarray(["train", "train", "train", "train", "val", "val"]),
        lead=np.asarray([1, 5, 10, 1, 5, 10], dtype=int),
        label=np.asarray([0, 1, 0, 1, 0, 1], dtype=int),
        episode_id=np.asarray([f"episode_{row}" for row in range(rows)]),
        origin_date=np.asarray(
            [f"2020-01-{row + 1:02d}" for row in range(rows)]
        ),
        mode_matrix=np.arange(rows * 9, dtype=float).reshape(rows, 9),
        target_path=np.zeros((rows, 10), dtype=float),
        har_prediction_path=np.zeros((rows, 10), dtype=float),
        prequential_residual_path=np.zeros((rows, 10), dtype=float),
        prequential_residual_valid=np.asarray(
            [True, True, True, True, False, False], dtype=bool
        ),
        mode_names=np.asarray([f"mode_{index}" for index in range(9)]),
    )


def test_loads_feature_archives_from_directory(tmp_path: Path) -> None:
    archive_path = (
        tmp_path / "feature_archives" / "ladder_symmetric_modes_fold_1.npz"
    )
    _write_fold_archive(archive_path, fold=1)

    loaded = load_ladder_feature_archives(tmp_path, folds=(1,))

    assert set(loaded) == {1}
    assert loaded[1]["mode_matrix"].shape == (6, 9)
    assert loaded[1]["target_path"].shape == (6, 10)


def test_loads_feature_archives_from_nested_zip(tmp_path: Path) -> None:
    source = tmp_path / "source" / "feature_archives"
    archive_path = source / "ladder_symmetric_modes_fold_1.npz"
    _write_fold_archive(archive_path, fold=1)
    zip_path = tmp_path / "ladder_readout_upgrade_seed20260721.zip"
    with zipfile.ZipFile(zip_path, "w") as bundle:
        bundle.write(
            archive_path,
            arcname=(
                "ladder_readout_upgrade_seed20260721/feature_archives/"
                "ladder_symmetric_modes_fold_1.npz"
            ),
        )

    loaded = load_ladder_feature_archives(zip_path, folds=(1,))

    assert set(loaded) == {1}
    assert loaded[1]["sample_id"][0].astype(str) == "sample_1_0"


def test_missing_requested_fold_fails(tmp_path: Path) -> None:
    archive_path = (
        tmp_path / "feature_archives" / "ladder_symmetric_modes_fold_1.npz"
    )
    _write_fold_archive(archive_path, fold=1)

    with pytest.raises(FileNotFoundError, match="missing folds"):
        load_ladder_feature_archives(tmp_path, folds=(1, 2))
