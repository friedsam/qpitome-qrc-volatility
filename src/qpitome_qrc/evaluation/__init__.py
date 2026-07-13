"""Stable public API for causal and prequential evaluation utilities."""

from .binary import (
    fit_feature_only_predict,
    fit_joint_predict,
    fit_offset_predict,
    logistic_pipeline,
)
from .historical_crossfit import historical_crossfit_d1_logits
from .residualization import (
    DEFAULT_N_SPLITS,
    d1_basis,
    fit_residualizer,
    residual_diagnostics,
    residualize_train_test,
)

__all__ = [
    "DEFAULT_N_SPLITS",
    "d1_basis",
    "fit_feature_only_predict",
    "fit_joint_predict",
    "fit_offset_predict",
    "fit_residualizer",
    "historical_crossfit_d1_logits",
    "logistic_pipeline",
    "residual_diagnostics",
    "residualize_train_test",
]
