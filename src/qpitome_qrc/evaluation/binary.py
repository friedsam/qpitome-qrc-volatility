"""Shared binary-classification model constructors and prediction helpers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.logistic_offset import fit_offset_logistic, sigmoid


def logistic_pipeline(C: float) -> Pipeline:
    """Return the standardized logistic model used by the locked assays."""

    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
    ])


def fit_offset_predict(
    H_train: np.ndarray,
    y: np.ndarray,
    H_test: np.ndarray,
    offset_train: np.ndarray,
    offset_test: float,
    l2: float,
) -> float:
    """Fit a standardized fixed-offset logistic correction and predict one row."""

    scaler = StandardScaler().fit(H_train)
    X_train = scaler.transform(H_train)
    X_test = scaler.transform(H_test)
    beta, intercept = fit_offset_logistic(X_train, y, offset_train, l2=l2)
    return float(sigmoid(np.array([offset_test + intercept + X_test[0] @ beta]))[0])
