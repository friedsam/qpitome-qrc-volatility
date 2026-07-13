"""Tests for fixed nonlinear protected path-input assay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from qpitome_qrc.day5.features import PROTECTED_FEATURE_BLOCKS, add_path_shape_features
from qpitome_qrc.evaluation.binary import (
    fit_offset_predict,
    fit_train_test_probability_arrays,
    logistic_pipeline,
)
from qpitome_qrc.evaluation.historical_crossfit import historical_crossfit_d1_logits
from qpitome_qrc.evaluation.residualization import d1_basis, residualize_train_test_safe
from qpitome_qrc.evaluation.scoring import proper_score_deltas

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_nonlinear_input_residual.py"
SPEC = importlib.util.spec_from_file_location("day5_nonlinear_input", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_nonlinear_maps_have_expected_shapes_and_are_deterministic() -> None:
    train = np.array([[0.0], [1.0], [2.0], [3.0]])
    test = np.array([[1.5], [2.5]])
    first = module.nonlinear_maps(train, test, seed=7)
    second = module.nonlinear_maps(train, test, seed=7)
    assert first["poly2"][0].shape == (4, 2)
    assert first["poly2"][1].shape == (2, 2)
    assert first["tanh32"][0].shape == (4, module.TANH_WIDTH)
    assert first["tanh32"][1].shape == (2, module.TANH_WIDTH)
    assert np.allclose(first["tanh32"][0], second["tanh32"][0])
    assert np.allclose(first["tanh32"][1], second["tanh32"][1])


def test_gaussian_control_matches_requested_shapes() -> None:
    train, test = module.gaussian_control((5, 3), (2, 3), seed=11)
    assert train.shape == (5, 3)
    assert test.shape == (2, 3)


def test_nonlinear_runner_uses_package_helpers_directly() -> None:
    assert module.FEATURE_BLOCKS is PROTECTED_FEATURE_BLOCKS
    assert module.add_path_shape_features is add_path_shape_features
    assert module.d1_basis is d1_basis
    assert module.residualize_train_test_safe is residualize_train_test_safe
    assert module.historical_crossfit_d1_logits is historical_crossfit_d1_logits
    assert module.fit_offset_predict is fit_offset_predict
    assert module.fit_train_test_probability_arrays is fit_train_test_probability_arrays
    assert module.logistic_pipeline is logistic_pipeline
    assert module.proper_score_deltas is proper_score_deltas
    assert not hasattr(module, "protected")
    assert not hasattr(module, "base")
    assert not hasattr(module, "crossfit")


def test_fixed_protocol_constants() -> None:
    assert module.OFFSET_L2 == 100.0
    assert module.RIDGE_ALPHA == 10.0
    assert module.TANH_WIDTH == 32
