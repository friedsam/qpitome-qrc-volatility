"""Stable public API for the locked day-5 branching experiments."""

from .features import feature_diagnostics, pca_block, split_blocks
from .protocol import (
    D1,
    EVAL_START,
    MIN_TRAIN,
    STATIC,
    add_extrema,
    differential_patterns,
    eligible_rows,
    load_frame,
    rydberg_config,
)

__all__ = [
    "D1",
    "EVAL_START",
    "MIN_TRAIN",
    "STATIC",
    "add_extrema",
    "differential_patterns",
    "eligible_rows",
    "feature_diagnostics",
    "load_frame",
    "pca_block",
    "rydberg_config",
    "split_blocks",
]
