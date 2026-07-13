"""Shared feature-block transforms for day-5 Rydberg assays."""

from __future__ import annotations

import numpy as np


def split_blocks(features: np.ndarray) -> dict[str, np.ndarray]:
    """Split occupation/pair observables and derive connected pair features."""

    occ = features[:, :10]
    raw_pairs = features[:, 10:]
    connected = np.empty_like(raw_pairs)
    k = 0
    for i in range(9):
        for j in range(i + 1, 10):
            connected[:, k] = raw_pairs[:, k] - occ[:, i] * occ[:, j]
            k += 1
    return {
        "occupations": occ,
        "raw_pairs": raw_pairs,
        "connected_pairs": connected,
        "all_raw": features,
        "occ_plus_connected": np.column_stack([occ, connected]),
    }
