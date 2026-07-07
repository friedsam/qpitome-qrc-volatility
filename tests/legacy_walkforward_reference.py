"""Frozen Phase 3 purged walk-forward geometry and alignment oracle."""

from __future__ import annotations

import pandas as pd

SPLITS = ("train", "val", "test")


def make_folds(
    n: int,
    *,
    n_folds: int,
    min_train: int,
    val_size: int,
    purge: int,
) -> list[dict]:
    first_test_start = min_train + val_size + purge
    if first_test_start >= n:
        raise ValueError("Not enough rows for requested fold construction")
    test_size = (n - first_test_start) // n_folds
    if test_size < 100:
        raise ValueError("Fold test size is too small")

    folds = []
    for i in range(n_folds):
        test_start = first_test_start + i * test_size
        test_end = n if i == n_folds - 1 else test_start + test_size
        val_end = test_start - purge
        val_start = val_end - val_size
        folds.append(
            {
                "fold": i + 1,
                "train": (0, val_start),
                "val": (val_start, val_end),
                "purge": (val_end, test_start),
                "test": (test_start, test_end),
            }
        )
    return folds


def aligned_frames(df: pd.DataFrame, fold: dict, lookback: int) -> dict[str, pd.DataFrame]:
    out = {}
    for split in SPLITS:
        a, b = fold[split]
        frame = df.iloc[a:b].copy().reset_index(drop=True)
        if len(frame) < lookback:
            raise ValueError(f"Fold {fold['fold']} {split} shorter than lookback")
        out[split] = frame.iloc[lookback - 1 :].reset_index(drop=True)
    return out
