"""Shared binary-classification model constructors and prediction helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from qpitome_qrc.baselines.logistic_offset import fit_offset_logistic, sigmoid


def logistic_pipeline(C: float) -> Pipeline:
    """Return the standardized logistic model used by the locked assays."""

    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
    ])


def transformed_logistic_pipeline(kind: str, C: float) -> Pipeline:
    """Return the locked linear, quadratic, or random-RBF logistic control."""

    steps: list[tuple[str, object]] = [("scale", StandardScaler())]
    if kind == "poly2":
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    elif kind == "rbf32":
        steps.append((
            "rbf",
            RBFSampler(gamma=0.25, n_components=32, random_state=17),
        ))
    elif kind != "linear":
        raise ValueError(kind)
    steps.append((
        "logit",
        LogisticRegression(C=C, max_iter=5000, solver="lbfgs"),
    ))
    return Pipeline(steps)


def fit_transformed_predict(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    kind: str,
    C: float,
) -> float:
    """Fit one locked transformed logistic control and predict one held-out row."""

    model = transformed_logistic_pipeline(kind, C)
    model.fit(X_train, y)
    return float(model.predict_proba(X_test)[0, 1])


def fit_feature_map_predict(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray],
    C: float,
) -> float:
    """Standardize inputs, apply a deterministic feature map, and predict one row."""

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train)
    scaled_test = scaler.transform(X_test)
    mapped_train = np.vstack([feature_map(row) for row in scaled_train])
    mapped_test = np.vstack([feature_map(row) for row in scaled_test])
    model = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    model.fit(mapped_train, y)
    return float(model.predict_proba(mapped_test)[0, 1])


def fit_joint_predict(
    d1_train: np.ndarray,
    H_train: np.ndarray,
    y: np.ndarray,
    d1_test: np.ndarray,
    H_test: np.ndarray,
    C: float,
) -> float:
    """Fit D1 plus a feature block and predict one held-out row."""

    model = logistic_pipeline(C)
    model.fit(np.column_stack([d1_train, H_train]), y)
    return float(model.predict_proba(np.column_stack([d1_test, H_test]))[0, 1])


def fit_feature_only_predict(
    H_train: np.ndarray,
    y: np.ndarray,
    H_test: np.ndarray,
    C: float,
) -> float:
    """Fit a feature-only standardized logistic model and predict one row."""

    model = logistic_pipeline(C)
    model.fit(H_train, y)
    return float(model.predict_proba(H_test)[0, 1])


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
