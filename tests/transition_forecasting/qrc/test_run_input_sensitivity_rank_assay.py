from __future__ import annotations

from pathlib import Path

import pytest

from scripts.transition_forecasting.qrc.run_input_sensitivity_rank_assay import (
    DEFAULT_FOLD_DIR,
    resolve_fold_dir,
)


def _write_fold_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "rematched_rolling_manifest.csv").write_text("sample_id\n")
    (path / "rematched_rolling_tensors.npz").write_bytes(b"placeholder")
    return path


def test_default_fold_dir_matches_restored_data_foundation() -> None:
    assert DEFAULT_FOLD_DIR == Path(
        "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
    )


def test_resolve_fold_dir_accepts_explicit_valid_directory(tmp_path: Path) -> None:
    expected = _write_fold_dir(tmp_path / "folds")

    assert resolve_fold_dir(expected) == expected


def test_resolve_fold_dir_reports_missing_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Could not locate"):
        resolve_fold_dir(Path("missing/folds"))
