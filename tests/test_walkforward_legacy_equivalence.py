"""Regression tests for the extracted Phase 3 walk-forward protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from qpitome_qrc.evaluation.walkforward import (
    align_fold_frames,
    make_purged_walkforward_folds,
)


LEGACY_REFERENCE = Path(__file__).with_name("legacy_walkforward_reference.py")


def load_legacy_module():
    spec = importlib.util.spec_from_file_location(
        "legacy_walkforward",
        LEGACY_REFERENCE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy reference: {LEGACY_REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "n_rows,n_folds,min_train,val_size,purge",
    [
        (7658, 5, 2500, 504, 60),
        (9000, 5, 2500, 504, 60),
        (5000, 3, 1800, 400, 20),
    ],
)
def test_fold_geometry_matches_frozen_legacy_oracle(
    n_rows: int,
    n_folds: int,
    min_train: int,
    val_size: int,
    purge: int,
) -> None:
    legacy = load_legacy_module()
    expected = legacy.make_folds(
        n_rows,
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
    )
    actual = make_purged_walkforward_folds(
        n_rows,
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
    )
    assert actual == expected


def test_alignment_matches_frozen_legacy_oracle() -> None:
    legacy = load_legacy_module()
    df = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=7658, freq="D"),
            "value": range(7658),
        }
    )
    fold = make_purged_walkforward_folds(
        len(df),
        n_folds=5,
        min_train=2500,
        val_size=504,
        purge=60,
    )[2]

    expected = legacy.aligned_frames(df, fold, lookback=40)
    actual = align_fold_frames(df, fold, lookback=40)

    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(actual[split], expected[split])


def test_alignment_matches_local_sequence_target_rows() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=7658, freq="D"),
            "value": range(7658),
        }
    )
    fold = make_purged_walkforward_folds(
        len(df),
        n_folds=5,
        min_train=2500,
        val_size=504,
        purge=60,
    )[0]
    aligned = align_fold_frames(df, fold, lookback=40)

    for split in ("train", "val", "test"):
        start, end = fold[split]
        assert aligned[split]["value"].iloc[0] == start + 39
        assert aligned[split]["value"].iloc[-1] == end - 1
        assert len(aligned[split]) == (end - start) - 39
