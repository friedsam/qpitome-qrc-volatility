"""Task-aligned residual correction utilities for branch-resolution models.

The classical baseline remains the primary predictor. A fixed reservoir is used
only to construct a small supervised representation of the baseline residual,
and a strongly regularized additive logit correction is fitted on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class OffsetCorrectionConfig:
    """Configuration for the convex additive-logit correction."""

    l2_penalty: float = 10.0
    max_iter: int = 100
    tolerance: float = 1e-10
    probability_clip: float = 1e-6


def logit_from_probability(p: np.ndarray, clip: float = 1e-6) -> np.ndarray:
    """Convert probabilities to finite logits."""
    q = np.clip(np.asarray(p, dtype=float), clip, 1.0 - clip)
    return np.log(q) - np.log1p(-q)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def fit_residual_pls(
    train_states: np.ndarray,
    train_residual: np.ndarray,
    test_state: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only PLS directions targeted at the baseline residual."""
    x_train = np.asarray(train_states, dtype=float)
    y_train = np.asarray(train_residual, dtype=float).reshape(-1, 1)
    x_test = np.asarray(test_state, dtype=float).reshape(1, -1)

    if x_train.ndim != 2:
        raise ValueError(f"Expected 2D train states; got {x_train.shape}")
    if len(x_train) != len(y_train):
        raise ValueError("Train-state and residual lengths differ")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    max_components = min(x_train.shape[0] - 1, x_train.shape[1])
    if n_components > max_components:
        raise ValueError(
            f"PLS n_components={n_components} exceeds train-only limit {max_components}"
        )
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("PLS inputs contain non-finite values")

    pls = PLSRegression(n_components=n_components, scale=True, max_iter=500)
    train_scores, _ = pls.fit_transform(x_train, y_train)
    test_scores = pls.transform(x_test)
    return np.asarray(train_scores, dtype=float), np.asarray(test_scores, dtype=float)


def fit_offset_logit_correction(
    train_scores: np.ndarray,
    y_train: np.ndarray,
    baseline_train_logit: np.ndarray,
    test_scores: np.ndarray,
    baseline_test_logit: float,
    config: OffsetCorrectionConfig | None = None,
) -> tuple[float, np.ndarray]:
    """Fit a no-intercept correction to frozen baseline logits.

    The optimization is convex. Newton updates are solved directly because the
    correction dimension is intentionally tiny (1-3 components).
    """
    cfg = config or OffsetCorrectionConfig()
    z = np.asarray(train_scores, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    offset = np.asarray(baseline_train_logit, dtype=float).reshape(-1)
    z_test = np.asarray(test_scores, dtype=float).reshape(1, -1)

    if z.ndim != 2:
        raise ValueError(f"Expected 2D train scores; got {z.shape}")
    if len(z) != len(y) or len(y) != len(offset):
        raise ValueError("Correction training lengths differ")
    if z_test.shape[1] != z.shape[1]:
        raise ValueError("Train/test correction dimensions differ")
    if cfg.l2_penalty <= 0:
        raise ValueError("l2_penalty must be positive")

    beta = np.zeros(z.shape[1], dtype=float)
    eye = np.eye(z.shape[1], dtype=float)

    for _ in range(cfg.max_iter):
        eta = offset + z @ beta
        p = sigmoid(eta)
        gradient = z.T @ (p - y) + cfg.l2_penalty * beta
        weights = np.clip(p * (1.0 - p), 1e-8, None)
        hessian = z.T @ (weights[:, None] * z) + cfg.l2_penalty * eye
        step = np.linalg.solve(hessian, gradient)
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < cfg.tolerance:
            beta = beta_new
            break
        beta = beta_new

    test_logit = float(baseline_test_logit + (z_test @ beta)[0])
    p_test = float(sigmoid(np.asarray([test_logit]))[0])
    p_test = float(np.clip(p_test, cfg.probability_clip, 1.0 - cfg.probability_clip))
    return p_test, beta
