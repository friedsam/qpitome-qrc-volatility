"""Tests for protected residual path-input audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5.features import PROTECTED_FEATURE_BLOCKS, add_path_shape_features
from qpitome_qrc.evaluation.binary import (
    fit_offset_predict,
    fit_train_test_probability_arrays,
    logistic_pipeline,
)
from qpitome_qrc.evaluation.historical_crossfit import historical_crossfit_d1_logits
from qpitome_qrc.evaluation.residualization import (
    d1_basis,
    residualize_train_test_safe,
)
from qpitome_qrc.evaluation.scoring import proper_score_deltas

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_protected_input_residual.py"
SPEC = importlib.util.spec_from_file_location("day5_protected_input", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_feature_blocks_are_fixed_and_unique() -> None:
    assert list(module.FEATURE_BLOCKS) == [
        "path_efficiency",
        "path_reversal_count",
        "path_early_late_imbalance",
        "trajectory_r_d1",
        "compact_four",
    ]
    assert len(module.FEATURE_BLOCKS["compact_four"]) == 4
    assert len(set(module.FEATURE_BLOCKS["compact_four"])) == 4
    assert module.FEATURE_BLOCKS is PROTECTED_FEATURE_BLOCKS


def test_score_deltas_returns_zero_for_identical_predictions() -> None:
    frame = pd.DataFrame({
        "y": [0, 1, 0, 1],
        "D1": [0.2, 0.8, 0.4, 0.6],
        "same": [0.2, 0.8, 0.4, 0.6],
        "cluster_id": ["a", "a", "b", "b"],
    })
    result = module.score_deltas(frame, "same")
    assert np.isclose(result["delta_logloss"], 0.0)
    assert np.isclose(result["delta_brier"], 0.0)
    assert np.isclose(result["cluster_mean_delta_logloss"], 0.0)
    assert np.isclose(result["cluster_mean_delta_brier"], 0.0)


def test_shape_safe_residualizer_preserves_single_column() -> None:
    rng = np.random.default_rng(12)
    X_train = rng.normal(size=(30, 4))
    H_train = rng.normal(size=(30, 1))
    X_test = rng.normal(size=(3, 4))
    H_test = rng.normal(size=(3, 1))

    R_train, R_test = residualize_train_test_safe(
        X_train,
        H_train,
        X_test,
        H_test,
        alpha=10.0,
    )

    assert R_train.shape == (30, 1)
    assert R_test.shape == (3, 1)


def test_protected_runner_uses_package_helpers_directly() -> None:
    assert module.add_path_shape_features is add_path_shape_features
    assert module.d1_basis is d1_basis
    assert module.residualize_train_test_safe is residualize_train_test_safe
    assert module.historical_crossfit_d1_logits is historical_crossfit_d1_logits
    assert module.fit_offset_predict is fit_offset_predict
    assert module.fit_train_test_probability_arrays is fit_train_test_probability_arrays
    assert module.logistic_pipeline is logistic_pipeline
    assert module.proper_score_deltas is proper_score_deltas
    assert not hasattr(module, "base")
    assert not hasattr(module, "crossfit")
    assert not hasattr(module, "input_audit")


def test_fixed_hyperparameters_match_confirmation_protocol() -> None:
    assert module.RIDGE_ALPHA == 10.0
    assert module.OFFSET_L2 == 100.0
