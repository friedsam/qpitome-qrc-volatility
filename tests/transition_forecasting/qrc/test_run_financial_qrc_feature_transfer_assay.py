from __future__ import annotations

from pathlib import Path

import pytest

from scripts.transition_forecasting.qrc.run_financial_qrc_feature_transfer_assay import (
    DEFAULT_FOLD_DIR,
    validate_fold_dir,
)


def test_default_fold_dir_uses_restored_data_foundation() -> None:
    assert DEFAULT_FOLD_DIR == Path(
        "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
    )


def test_validate_fold_dir_requires_manifest_and_tensor(tmp_path: Path) -> None:
    fold_dir = tmp_path / "folds"
    fold_dir.mkdir()
    (fold_dir / "rematched_rolling_manifest.csv").write_text("sample_id\n")

    with pytest.raises(FileNotFoundError, match="rematched_rolling_tensors.npz"):
        validate_fold_dir(fold_dir)

    (fold_dir / "rematched_rolling_tensors.npz").write_bytes(b"placeholder")
    assert validate_fold_dir(fold_dir) == fold_dir
