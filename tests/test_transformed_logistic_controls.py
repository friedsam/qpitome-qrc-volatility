from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from qpitome_qrc.evaluation.binary import (
    fit_transformed_predict,
    transformed_logistic_pipeline,
)


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transformed_logistic_pipeline_has_locked_steps() -> None:
    assert list(transformed_logistic_pipeline("linear", 1.0).named_steps) == [
        "scale",
        "logit",
    ]
    assert list(transformed_logistic_pipeline("poly2", 0.05).named_steps) == [
        "scale",
        "poly",
        "logit",
    ]
    assert list(transformed_logistic_pipeline("rbf32", 0.05).named_steps) == [
        "scale",
        "rbf",
        "logit",
    ]


def test_fit_transformed_predict_matches_locked_rbf_sequence() -> None:
    X_train = np.array([
        [-2.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.5],
        [2.0, 1.5],
        [3.0, -0.5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[0.75, 0.25]])

    actual = fit_transformed_predict(
        X_train,
        y,
        X_test,
        kind="rbf32",
        C=0.05,
    )

    expected_model = Pipeline([
        ("scale", StandardScaler()),
        ("rbf", RBFSampler(gamma=0.25, n_components=32, random_state=17)),
        ("logit", LogisticRegression(C=0.05, max_iter=5000, solver="lbfgs")),
    ])
    expected_model.fit(X_train, y)
    expected = float(expected_model.predict_proba(X_test)[0, 1])

    np.testing.assert_allclose(actual, expected)


def test_regularized_control_script_reexports_shared_helpers() -> None:
    module = load_script(
        "regularized_control_helpers",
        "scripts/modeling/run_cross_market_day5_regularized_controls.py",
    )

    assert module.make_model is transformed_logistic_pipeline
    assert module.transformed_logistic_pipeline is transformed_logistic_pipeline
    assert module.fit_transformed_predict is fit_transformed_predict
