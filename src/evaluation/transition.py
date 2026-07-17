"""Metrics for signed volatility-transition forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_transition_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return RMSE and R-squared for a transition forecast."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")

    mask = np.isfinite(true) & np.isfinite(pred)
    if int(mask.sum()) < 3:
        raise ValueError("Need at least 3 finite observations for transition metrics.")

    true = true[mask]
    pred = pred[mask]
    return {
        "innovation_rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "innovation_r2": float(r2_score(true, pred)),
    }
