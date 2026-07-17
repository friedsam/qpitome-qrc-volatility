"""Shared residualization utilities for held-out feature corrections."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEFAULT_N_SPLITS = 5


def ridge_pipeline(alpha: float = 10.0) -> Pipeline:
    """Return the standardized multi-output Ridge model used by Day-5 audits."""

    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def d1_basis(X: np.ndarray, kind: str) -> np.ndarray:
    """Construct the locked linear or quadratic D1 residualization basis."""

    X = np.asarray(X, dtype=float)
    if kind == "linear":
        return X
    if kind == "quadratic":
        return PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    raise ValueError(f"Unknown residualizer kind: {kind}")


def fit_residualizer(
    X_train: np.ndarray,
    H_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted train and test values from standardized multi-output Ridge."""

    x_scaler = StandardScaler().fit(X_train)
    h_scaler = StandardScaler().fit(H_train)
    Xs = x_scaler.transform(X_train)
    Xts = x_scaler.transform(X_test)
    Hs = h_scaler.transform(H_train)
    model = Ridge(alpha=alpha).fit(Xs, Hs)
    fitted_train = h_scaler.inverse_transform(model.predict(Xs))
    fitted_test = h_scaler.inverse_transform(model.predict(Xts))
    return fitted_train, fitted_test


def residualize_train_test(
    X_train: np.ndarray,
    H_train: np.ndarray,
    X_test: np.ndarray,
    H_test: np.ndarray,
    alpha: float,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fit train residuals and residualize test rows using all training rows."""

    splits = min(n_splits, len(X_train))
    if splits < 2:
        raise ValueError("At least two training rows are required")

    predicted_train = np.empty_like(H_train, dtype=float)
    kfold = KFold(n_splits=splits, shuffle=False)
    for fit_idx, valid_idx in kfold.split(X_train):
        _, predicted_valid = fit_residualizer(
            X_train[fit_idx], H_train[fit_idx], X_train[valid_idx], alpha
        )
        predicted_train[valid_idx] = predicted_valid

    _, predicted_test = fit_residualizer(X_train, H_train, X_test, alpha)
    return H_train - predicted_train, H_test - predicted_test


def residualize_train_test_safe(
    X_train: np.ndarray,
    H_train: np.ndarray,
    X_test: np.ndarray,
    H_test: np.ndarray,
    alpha: float,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Residualize while preserving two-dimensional single-output feature blocks."""

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    H_train = np.asarray(H_train, dtype=float)
    H_test = np.asarray(H_test, dtype=float)
    if H_train.ndim == 1:
        H_train = H_train.reshape(-1, 1)
    if H_test.ndim == 1:
        H_test = H_test.reshape(-1, 1)

    def fitted_values(
        X_fit: np.ndarray,
        H_fit: np.ndarray,
        X_eval: np.ndarray,
    ) -> np.ndarray:
        x_scaler = StandardScaler().fit(X_fit)
        h_scaler = StandardScaler().fit(H_fit)
        Xs = x_scaler.transform(X_fit)
        Xes = x_scaler.transform(X_eval)
        Hs = h_scaler.transform(H_fit)
        model = Ridge(alpha=alpha).fit(Xs, Hs)
        predicted = np.asarray(model.predict(Xes), dtype=float)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)
        return h_scaler.inverse_transform(predicted)

    splits = min(n_splits, len(X_train))
    if splits < 2:
        raise ValueError("At least two training rows are required")
    predicted_train = np.empty_like(H_train, dtype=float)
    kfold = KFold(n_splits=splits, shuffle=False)
    for fit_idx, valid_idx in kfold.split(X_train):
        predicted_train[valid_idx] = fitted_values(
            X_train[fit_idx], H_train[fit_idx], X_train[valid_idx]
        )
    predicted_test = fitted_values(X_train, H_train, X_test)
    return H_train - predicted_train, H_test - predicted_test


def residual_diagnostics(H: np.ndarray, R: np.ndarray) -> dict[str, float]:
    """Summarize variance removed and residual scale."""

    total = float(np.sum((H - H.mean(axis=0, keepdims=True)) ** 2))
    residual = float(np.sum(R**2))
    explained = 1.0 - residual / total if total > 1e-15 else float("nan")
    return {
        "fraction_output_variance_removed": float(explained),
        "residual_rms": float(np.sqrt(np.mean(R**2))),
    }
