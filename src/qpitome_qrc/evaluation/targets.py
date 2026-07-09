"""Shared target and residual utilities for Phase 3.

This module owns target algebra only. Model fitting, feature selection, fold
construction, and reporting remain in experiment runners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_CURRENT_RV = "rv_20d"
DEFAULT_FUTURE_RV = "future_rv_20d"
DEFAULT_INNOVATION_TARGET = "rv_innovation_20d"


def add_log_rv_innovation(
    df: pd.DataFrame,
    *,
    current_col: str = DEFAULT_CURRENT_RV,
    future_col: str = DEFAULT_FUTURE_RV,
    target_col: str = DEFAULT_INNOVATION_TARGET,
) -> pd.DataFrame:
    """Return a copy with ``log(future_rv / current_rv)`` added.

    Persistence / zero-change is exactly zero on this target.
    """
    missing = {current_col, future_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing RV columns: {sorted(missing)}")

    current = df[current_col].to_numpy(dtype=float)
    future = df[future_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(current)) or np.any(~np.isfinite(future)):
        raise ValueError("RV columns contain non-finite values")
    if np.any(current <= 0.0) or np.any(future <= 0.0):
        raise ValueError("RV columns must be strictly positive")

    out = df.copy()
    out[target_col] = np.log(future / current)
    return out


def reconstruct_future_rv(
    current_rv: np.ndarray | pd.Series,
    innovation_prediction: np.ndarray | pd.Series,
) -> np.ndarray:
    """Reconstruct future RV from current RV and a log-innovation prediction."""
    current = np.asarray(current_rv, dtype=float)
    innovation = np.asarray(innovation_prediction, dtype=float)
    if current.shape != innovation.shape:
        raise ValueError(
            "current_rv and innovation_prediction must have identical shapes"
        )
    if np.any(current <= 0.0):
        raise ValueError("current_rv must be strictly positive")
    return current * np.exp(innovation)


def residual_from_oof_prediction(
    y_true: np.ndarray | pd.Series,
    y_oof_prediction: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return residuals using out-of-fold predictions only.

    The function is deliberately algebraic. The caller is responsible for
    proving that ``y_oof_prediction`` is genuinely out of fold.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_oof_prediction, dtype=float)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_oof_prediction must have identical shapes")
    if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(prediction)):
        raise ValueError("Residual inputs contain non-finite values")
    return truth - prediction


def add_residual_column(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    residual_col: str = "residual",
) -> pd.DataFrame:
    """Return a copy with ``target - prediction`` added."""
    missing = {target_col, prediction_col} - set(frame.columns)
    if missing:
        raise KeyError(f"Missing residual columns: {sorted(missing)}")

    out = frame.copy()
    out[residual_col] = residual_from_oof_prediction(
        out[target_col], out[prediction_col]
    )
    return out
