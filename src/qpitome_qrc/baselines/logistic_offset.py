"""Shared logistic-offset utilities for incremental classification models."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EPS = 1e-6


def clip_prob(p: np.ndarray | float) -> np.ndarray:
    """Clip probabilities away from exact zero and one."""
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p: np.ndarray | float) -> np.ndarray:
    """Stable logit transform after probability clipping."""
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def fit_offset_logistic(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, float]:
    """Fit ``sigmoid(offset + intercept + X @ beta)`` with L2 on ``beta``.

    ``X`` must already be standardized. The intercept is unpenalized.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if y.ndim != 1 or offset.ndim != 1:
        raise ValueError("y and offset must be one-dimensional")
    if len(X) != len(y) or len(y) != len(offset):
        raise ValueError("X, y, and offset must have matching row counts")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    n_features = X.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = theta[0]
        beta = theta[1:]
        eta = offset + intercept + X @ beta
        p = clip_prob(sigmoid(eta))
        loss = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        loss += 0.5 * l2 * float(beta @ beta)

        residual = p - y
        gradient = np.concatenate(
            [[float(np.sum(residual))], X.T @ residual + l2 * beta]
        )
        return float(loss), gradient

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=np.zeros(n_features + 1),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Offset optimization failed: {result.message}")

    return result.x[1:], float(result.x[0])
