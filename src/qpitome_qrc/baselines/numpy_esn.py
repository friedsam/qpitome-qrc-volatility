"""Deterministic NumPy echo-state reservoir used by the Phase 2/3 benchmarks.

This module owns only the reservoir mechanics and log-target ridge readout.
Evaluation geometry, preprocessing, model selection, and reporting belong to
the experiment runners.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SPLIT_NAMES = ("train", "val", "test")


def spectral_scale(W: np.ndarray, radius: float) -> np.ndarray:
    """Rescale a recurrent matrix to the requested spectral radius."""
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W * (radius / max(rho, 1e-12))


def make_esn_weights(
    n_inputs: int,
    n_reservoir: int,
    spectral_radius: float,
    input_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the historical deterministic NumPy ESN input/recurrent weights."""
    rng = np.random.default_rng(seed)

    W_in = rng.normal(
        0.0,
        input_scale,
        size=(n_reservoir, n_inputs),
    )

    W = rng.normal(
        0.0,
        1.0,
        size=(n_reservoir, n_reservoir),
    )

    W *= rng.random(W.shape) < 0.10

    return W_in, spectral_scale(W, spectral_radius)


def esn_states(
    X: np.ndarray,
    W_in: np.ndarray,
    W: np.ndarray,
    leak: float,
) -> np.ndarray:
    """Convert local input windows into final-state-plus-final-input features.

    The reservoir state is reset to zero for every input window. This preserves
    the exact historical NumPy ESN behavior.
    """
    rows = []

    for window in X:
        h = np.zeros(W.shape[0])

        for u_t in window:
            h_new = np.tanh(W_in @ u_t + W @ h)
            h = (1.0 - leak) * h + leak * h_new

        rows.append(np.concatenate([h, window[-1]]))

    return np.asarray(rows)


def fit_log_ridge_scores(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    """Fit the historical standardized ridge readout on log volatility."""
    scaler = StandardScaler()
    model = Ridge(alpha=alpha)

    model.fit(
        scaler.fit_transform(features["train"]),
        np.log(np.maximum(targets["train"], 1e-8)),
    )

    return {
        split: model.predict(scaler.transform(features[split]))
        for split in SPLIT_NAMES
    }
