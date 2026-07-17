"""Deterministic ridge-logistic readouts with optional fixed logit offsets."""

from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    penalty: float,
    offset: np.ndarray | None = None,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x must be 2D and y must be a same-length 1D array")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")

    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=float)
    prior = np.clip(float(y.mean()), 1e-5, 1.0 - 1e-5)
    beta[0] = np.log(prior / (1.0 - prior))
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    fixed = np.zeros(len(y), dtype=float) if offset is None else np.asarray(offset, dtype=float)
    if fixed.shape != y.shape:
        raise ValueError("offset must have the same shape as y")

    for _ in range(max_iter):
        probability = sigmoid(fixed + design @ beta)
        weight = np.clip(probability * (1.0 - probability), 1e-7, None)
        gradient = design.T @ (probability - y) + regularizer @ beta
        hessian = design.T @ (weight[:, None] * design) + regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        updated = beta - step
        if float(np.max(np.abs(updated - beta))) < tolerance:
            beta = updated
            break
        beta = updated
    return beta


def standardize_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    if train_x.ndim != 2:
        raise ValueError("train_x must be 2D")
    test_was_1d = test_x.ndim == 1
    test_matrix = test_x[None, :] if test_was_1d else test_x
    if test_matrix.ndim != 2 or test_matrix.shape[1] != train_x.shape[1]:
        raise ValueError("test_x must have the same feature dimension as train_x")

    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized_train = (train_x - mean) / scale
    standardized_test = (test_matrix - mean) / scale
    if test_was_1d:
        standardized_test = standardized_test[0]
    return standardized_train, standardized_test, mean, scale


def fit_predict_probability(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    penalty: float,
    train_offset: np.ndarray | None = None,
    test_offset: float | np.ndarray | None = None,
) -> float | np.ndarray:
    standardized_train, standardized_test, _, _ = standardize_train_test(train_x, test_x)
    beta = fit_ridge_logistic(
        standardized_train,
        train_y,
        penalty=penalty,
        offset=train_offset,
    )
    test_matrix = standardized_test[None, :] if standardized_test.ndim == 1 else standardized_test
    fixed = 0.0 if test_offset is None else np.asarray(test_offset, dtype=float)
    probability = sigmoid(fixed + np.column_stack([np.ones(len(test_matrix)), test_matrix]) @ beta)
    return float(probability[0]) if standardized_test.ndim == 1 else probability
