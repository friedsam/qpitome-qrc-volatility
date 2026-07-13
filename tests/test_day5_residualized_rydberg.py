"""Tests for the D1-residualized Rydberg diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from qpitome_qrc.evaluation.binary import (
    fit_train_test_probability_arrays,
    logistic_pipeline,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_residualized_rydberg_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_residualized_rydberg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_d1_basis_shapes() -> None:
    X = np.arange(20.0).reshape(5, 4)
    assert module.d1_basis(X, "linear").shape == (5, 4)
    assert module.d1_basis(X, "quadratic").shape == (5, 14)


def test_residualization_removes_linear_mapping() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 4))
    weights = rng.normal(size=(4, 6))
    H = X @ weights
    X_test = rng.normal(size=(10, 4))
    H_test = X_test @ weights
    R_train, R_test = module.residualize_train_test(X, H, X_test, H_test, alpha=1e-8)
    assert np.sqrt(np.mean(R_train**2)) < 1e-6
    assert np.sqrt(np.mean(R_test**2)) < 1e-6


def test_quadratic_basis_removes_quadratic_mapping() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 4))
    Xq = module.d1_basis(X, "quadratic")
    weights = rng.normal(size=(Xq.shape[1], 5))
    H = Xq @ weights
    X_test = rng.normal(size=(12, 4))
    Xq_test = module.d1_basis(X_test, "quadratic")
    H_test = Xq_test @ weights
    R_train, R_test = module.residualize_train_test(Xq, H, Xq_test, H_test, alpha=1e-8)
    assert np.sqrt(np.mean(R_train**2)) < 1e-5
    assert np.sqrt(np.mean(R_test**2)) < 1e-5


def test_residual_diagnostics_reports_removed_variance() -> None:
    H = np.arange(40.0).reshape(10, 4)
    R = np.zeros_like(H)
    diagnostic = module.residual_diagnostics(H, R)
    assert diagnostic["fraction_output_variance_removed"] == 1.0
    assert diagnostic["residual_rms"] == 0.0


def test_grouped_probability_helper_matches_locked_sequence() -> None:
    X_train = np.array([
        [-2.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.5],
        [2.0, 1.5],
        [3.0, -0.5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([
        [0.75, 0.25],
        [1.25, -0.25],
        [2.25, 0.75],
    ])

    train_probabilities, test_probabilities = fit_train_test_probability_arrays(
        X_train,
        y,
        X_test,
        C=1.0,
    )

    model = logistic_pipeline(1.0)
    model.fit(X_train, y)
    np.testing.assert_allclose(
        train_probabilities,
        model.predict_proba(X_train)[:, 1],
    )
    np.testing.assert_allclose(
        test_probabilities,
        model.predict_proba(X_test)[:, 1],
    )
    assert module.fit_train_test_probability_arrays is fit_train_test_probability_arrays
