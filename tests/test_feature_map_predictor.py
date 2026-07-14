from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.binary import fit_feature_map_predict


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polynomial_map(row: np.ndarray) -> np.ndarray:
    return np.concatenate([row, row**2])


def test_feature_map_predictor_matches_locked_manual_sequence() -> None:
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

    actual = fit_feature_map_predict(
        X_train,
        y,
        X_test,
        polynomial_map,
        C=0.1,
    )

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train)
    scaled_test = scaler.transform(X_test)
    mapped_train = np.vstack([polynomial_map(row) for row in scaled_train])
    mapped_test = np.vstack([polynomial_map(row) for row in scaled_test])
    model = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs")
    model.fit(mapped_train, y)
    expected = float(model.predict_proba(mapped_test)[0, 1])

    np.testing.assert_allclose(actual, expected)


def test_fixed_rydberg_script_reexports_feature_map_predictor() -> None:
    module = load_script(
        "fixed_rydberg_feature_map_predictor",
        "scripts/modeling/day5_branching/static_rydberg/run_cross_market_day5_fixed_rydberg_feature.py",
    )

    assert module.fit_feature_map_predict is fit_feature_map_predict
