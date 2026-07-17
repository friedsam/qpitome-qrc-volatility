"""Shared empirical-prior baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_empirical_prior(
    train: pd.DataFrame,
    *,
    target: str,
    group_column: str | None = None,
    group_value: object | None = None,
    min_group_rows: int = 5,
    clip: float = 1e-6,
) -> float:
    """Return a clipped pooled or group-specific empirical event rate.

    Group-specific estimates fall back to the pooled training frame when the
    requested group has fewer than ``min_group_rows`` observations.
    """

    use = train
    if group_column is not None:
        use = train[train[group_column] == group_value]
        if len(use) < min_group_rows:
            use = train
    return float(np.clip(use[target].mean(), clip, 1.0 - clip))
