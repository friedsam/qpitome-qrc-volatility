"""Stable public API for causal and prequential evaluation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .binary import (
    fit_feature_map_predict,
    fit_feature_only_predict,
    fit_joint_predict,
    fit_offset_predict,
    fit_train_test_probabilities,
    fit_transformed_predict,
    logistic_pipeline,
    transformed_logistic_pipeline,
)
from .residualization import (
    DEFAULT_N_SPLITS,
    d1_basis,
    fit_residualizer,
    residual_diagnostics,
    residualize_train_test,
    ridge_pipeline,
)
from .scoring import binary_summary, cluster_weighted_summary, proper_score_deltas

if TYPE_CHECKING:
    from .historical_crossfit import historical_crossfit_d1_logits

__all__ = [
    "DEFAULT_N_SPLITS",
    "binary_summary",
    "cluster_weighted_summary",
    "d1_basis",
    "fit_feature_map_predict",
    "fit_feature_only_predict",
    "fit_joint_predict",
    "fit_offset_predict",
    "fit_residualizer",
    "fit_train_test_probabilities",
    "fit_transformed_predict",
    "historical_crossfit_d1_logits",
    "logistic_pipeline",
    "proper_score_deltas",
    "residual_diagnostics",
    "residualize_train_test",
    "ridge_pipeline",
    "transformed_logistic_pipeline",
]


def __getattr__(name: str) -> Any:
    """Lazily expose helpers that depend on the Day-5 protocol package."""

    if name == "historical_crossfit_d1_logits":
        from .historical_crossfit import historical_crossfit_d1_logits

        return historical_crossfit_d1_logits
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
