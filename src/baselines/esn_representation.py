"""Input transforms used by the frozen classical ESN comparator."""
from __future__ import annotations

import numpy as np

INPUT_REPRESENTATIONS = ("level", "level_diff", "level_diff_time")


def transform_input(X: np.ndarray, representation: str) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 1:
        raise ValueError(f"expected X shaped (samples, steps, 1), got {values.shape}")
    level = values[..., 0]
    difference = np.diff(level, axis=1, prepend=level[:, :1])
    time = np.repeat(
        np.linspace(0.0, 1.0, level.shape[1], dtype=float)[None, :],
        level.shape[0],
        axis=0,
    )
    if representation == "level":
        return level[..., None]
    if representation == "level_diff":
        return np.stack([level, difference], axis=-1)
    if representation == "level_diff_time":
        return np.stack([level, difference, time], axis=-1)
    raise ValueError(f"unknown input representation: {representation}")
