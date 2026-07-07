"""Canonical target definitions for Phase 3 volatility-transition experiments.

The primary transition target is a signed log ratio between future and current
20-day realized volatility. It removes the trivial level-tracking advantage of
highly persistent overlapping volatility windows while preserving the original
forecast horizon and dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RV_LEVEL_TARGET = "future_rv_20d"
RV_REFERENCE_COLUMN = "rv_20d"
RV_INNOVATION_TARGET = "rv_innovation_20d"
_EPS = 1e-12


def add_rv_innovation_target(
    frame: pd.DataFrame,
    *,
    future_column: str = RV_LEVEL_TARGET,
    reference_column: str = RV_REFERENCE_COLUMN,
    target_column: str = RV_INNOVATION_TARGET,
    eps: float = _EPS,
    copy: bool = True,
) -> pd.DataFrame:
    """Add ``log(future_rv / current_rv)`` as a signed transition target.

    Positive values indicate increasing volatility over the forecast horizon;
    negative values indicate decreasing volatility. The input frame is copied
    by default so callers cannot silently mutate canonical datasets.
    """

    required = [future_column, reference_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns required for innovation target: {missing}")

    result = frame.copy() if copy else frame
    future = result[future_column].to_numpy(dtype=float)
    reference = result[reference_column].to_numpy(dtype=float)

    if np.any(~np.isfinite(future)) or np.any(~np.isfinite(reference)):
        raise ValueError("Innovation target inputs must be finite before target construction.")
    if np.any(future <= 0.0) or np.any(reference <= 0.0):
        raise ValueError("Innovation target inputs must be strictly positive.")

    result[target_column] = np.log(
        np.maximum(future, eps) / np.maximum(reference, eps)
    )
    return result


def reconstruct_future_rv(
    reference_rv: np.ndarray,
    innovation_prediction: np.ndarray,
) -> np.ndarray:
    """Reconstruct positive future RV from current RV and predicted innovation."""

    reference = np.asarray(reference_rv, dtype=float)
    innovation = np.asarray(innovation_prediction, dtype=float)
    if reference.shape != innovation.shape:
        raise ValueError(
            "reference_rv and innovation_prediction must have identical shapes."
        )
    if np.any(~np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError("reference_rv must be finite and strictly positive.")
    if np.any(~np.isfinite(innovation)):
        raise ValueError("innovation_prediction must be finite.")

    return reference * np.exp(innovation)
