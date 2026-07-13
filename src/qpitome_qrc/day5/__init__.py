"""Stable public API for the locked day-5 branching experiments."""

from .features import (
    PATH_SHAPE_BLOCKS,
    add_path_shape_features,
    feature_diagnostics,
    pca_block,
    split_blocks,
)
from .protocol import (
    D1,
    EVAL_START,
    MIN_TRAIN,
    STATIC,
    add_extrema,
    attach_cluster_start,
    differential_patterns,
    eligible_rows,
    load_frame,
    rydberg_config,
)

__all__ = [
    "D1",
    "EVAL_START",
    "MIN_TRAIN",
    "PATH_SHAPE_BLOCKS",
    "STATIC",
    "add_extrema",
    "add_path_shape_features",
    "attach_cluster_start",
    "differential_patterns",
    "eligible_rows",
    "feature_diagnostics",
    "load_frame",
    "pca_block",
    "rydberg_config",
    "split_blocks",
]
