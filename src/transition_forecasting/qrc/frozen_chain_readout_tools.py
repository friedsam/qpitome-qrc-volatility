"""Exact chronological inner split required by the canonical Case151 assay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.representation_screen_analysis import _parse_mixed_utc


def chronological_inner_split(
    origin_date: np.ndarray,
    eligible: np.ndarray,
    *,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"origin_date": np.asarray(origin_date).astype(str)})
    dates = _parse_mixed_utc(frame["origin_date"])
    mask = np.asarray(eligible, dtype=bool)
    unique_dates = np.asarray(sorted(dates[mask].unique()))
    if len(unique_dates) < 4:
        raise ValueError("not enough unique training dates for an inner split")
    holdout_dates = max(1, int(np.ceil(len(unique_dates) * holdout_fraction)))
    holdout_dates = min(holdout_dates, len(unique_dates) - 2)
    cutoff = unique_dates[-holdout_dates]
    inner_fit = mask & dates.lt(cutoff).to_numpy()
    inner_tune = mask & dates.ge(cutoff).to_numpy()
    if inner_fit.sum() < 10 or inner_tune.sum() < 5:
        raise ValueError(
            "inner split is too small: "
            f"fit={int(inner_fit.sum())}, tune={int(inner_tune.sum())}"
        )
    return inner_fit, inner_tune
