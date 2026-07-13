"""Shared binary-classification model constructors."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic_pipeline(C: float) -> Pipeline:
    """Return the standardized logistic model used by the locked assays."""

    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
    ])
