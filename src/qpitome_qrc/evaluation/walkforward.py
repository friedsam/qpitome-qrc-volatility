"""Canonical chronological evaluation geometry for Phase 3.

This module owns only the shared forecasting protocol:

- expanding training history;
- fixed-size validation block immediately before a purge gap;
- non-overlapping chronological test blocks;
- optional common-date alignment for models with a rolling lookback.

Model-specific hyperparameters and feature construction do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PurgedWalkForwardConfig:
    """Geometry of an expanding purged walk-forward evaluation."""

    n_folds: int = 5
    min_train: int = 2500
    val_size: int = 504
    purge: int = 60
    min_test_size: int = 100

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError("n_folds must be positive")
        if self.min_train < 1:
            raise ValueError("min_train must be positive")
        if self.val_size < 1:
            raise ValueError("val_size must be positive")
        if self.purge < 0:
            raise ValueError("purge must be non-negative")
        if self.min_test_size < 1:
            raise ValueError("min_test_size must be positive")


def make_purged_walkforward_folds(
    n_rows: int,
    *,
    n_folds: int,
    min_train: int,
    val_size: int,
    purge: int,
    min_test_size: int = 100,
) -> list[dict[str, int | tuple[int, int]]]:
    """Build the Phase 3 expanding purged walk-forward fold geometry.

    Each fold has the form::

        train | validation | purge | test

    Training always starts at row zero and expands across folds. Validation has
    fixed width. Test blocks are chronological and non-overlapping; the final
    fold receives any integer-division remainder.
    """

    config = PurgedWalkForwardConfig(
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
        min_test_size=min_test_size,
    )
    first_test_start = config.min_train + config.val_size + config.purge
    if first_test_start >= n_rows:
        raise ValueError("Not enough rows for requested fold construction")

    test_size = (n_rows - first_test_start) // config.n_folds
    if test_size < config.min_test_size:
        raise ValueError(f"test_size too small: {test_size}")

    folds: list[dict[str, int | tuple[int, int]]] = []
    for i in range(config.n_folds):
        test_start = first_test_start + i * test_size
        test_end = n_rows if i == config.n_folds - 1 else test_start + test_size
        val_end = test_start - config.purge
        val_start = val_end - config.val_size
        train_end = val_start
        if train_end < config.min_train:
            raise ValueError("Invalid fold construction")

        folds.append(
            {
                "fold": i + 1,
                "train": (0, train_end),
                "val": (val_start, val_end),
                "purge": (val_end, test_start),
                "test": (test_start, test_end),
            }
        )
    return folds


def slice_fold_frames(
    df: pd.DataFrame,
    fold: Mapping[str, int | tuple[int, int]],
) -> dict[str, pd.DataFrame]:
    """Return reset-index train/validation/test frames for one fold."""

    out: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        start, end = fold[split]  # type: ignore[misc]
        out[split] = df.iloc[start:end].copy().reset_index(drop=True)
    return out


def align_fold_frames(
    df: pd.DataFrame,
    fold: Mapping[str, int | tuple[int, int]],
    lookback: int,
) -> dict[str, pd.DataFrame]:
    """Align tabular models to the target dates produced by local sequences.

    A rolling sequence of length ``lookback`` built independently inside each
    split produces its first target at local row ``lookback - 1``. This helper
    drops exactly those unavailable leading rows from tabular models so all
    model families are evaluated on identical dates without double-applying the
    lookback.
    """

    if lookback < 1:
        raise ValueError("lookback must be positive")

    frames = slice_fold_frames(df, fold)
    aligned: dict[str, pd.DataFrame] = {}
    fold_id = fold.get("fold", "?")
    for split, frame in frames.items():
        if len(frame) < lookback:
            raise ValueError(f"Fold {fold_id} {split} shorter than lookback")
        aligned[split] = frame.iloc[lookback - 1 :].reset_index(drop=True)
    return aligned
