from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.metrics import ClassificationMetrics, evaluate_binary_classifier

ReadoutKind = Literal["logistic", "ridge", "mlp"]


@dataclass(frozen=True)
class ReadoutConfig:
    """Trainable readout configuration for reservoir features."""

    kind: ReadoutKind = "logistic"
    scale_states: bool = False
    class_weight: str | None = "balanced"
    random_state: int = 42
    params: dict[str, Any] | None = None


def build_readout(config: ReadoutConfig) -> BaseEstimator:
    """Build a sklearn-compatible readout.

    Logistic/ridge are linear readouts. MLP is a nonlinear hybrid reservoir-DNN
    readout and should be reported separately from pure ESN results.
    """
    params = dict(config.params or {})

    if config.kind == "logistic":
        model = LogisticRegression(
            class_weight=config.class_weight,
            max_iter=3000,
            random_state=config.random_state,
            **params,
        )
    elif config.kind == "ridge":
        model = RidgeClassifier(
            class_weight=config.class_weight,
            random_state=config.random_state,
            **params,
        )
    elif config.kind == "mlp":
        # sklearn MLPClassifier does not support class_weight directly.
        # Use small networks and validation metrics to control overfitting.
        model = MLPClassifier(
            hidden_layer_sizes=params.pop("hidden_layer_sizes", (32,)),
            activation=params.pop("activation", "relu"),
            alpha=params.pop("alpha", 1e-3),
            learning_rate_init=params.pop("learning_rate_init", 1e-3),
            max_iter=params.pop("max_iter", 500),
            early_stopping=params.pop("early_stopping", True),
            validation_fraction=params.pop("validation_fraction", 0.15),
            random_state=config.random_state,
            **params,
        )
    else:
        raise ValueError(f"Unknown readout kind: {config.kind}")

    if config.scale_states:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def readout_scores(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Return positive-class scores for heterogeneous readouts."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    raise TypeError(f"Readout does not expose predict_proba or decision_function: {type(model)}")


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, thresholds=None) -> tuple[float, float]:
    """Choose threshold maximizing validation F1 for class 1."""
    if thresholds is None:
        thresholds = np.linspace(np.nanmin(y_score), np.nanmax(y_score), 101)

    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


@dataclass
class ReadoutFitResult:
    readout: BaseEstimator
    threshold: float
    val_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    val_scores: np.ndarray
    test_scores: np.ndarray


def fit_readout(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_val: np.ndarray,
    y_val: np.ndarray,
    H_test: np.ndarray,
    y_test: np.ndarray,
    config: ReadoutConfig,
    tune_threshold: bool = True,
) -> ReadoutFitResult:
    """Fit readout on reservoir features and evaluate val/test."""
    readout = build_readout(config)
    readout.fit(H_train, y_train)

    val_scores = readout_scores(readout, H_val)
    test_scores = readout_scores(readout, H_test)

    threshold = find_best_threshold(y_val, val_scores)[0] if tune_threshold else 0.5
    val_metrics = evaluate_binary_classifier(y_val, val_scores, threshold=threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_scores, threshold=threshold)

    return ReadoutFitResult(
        readout=readout,
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_scores=val_scores,
        test_scores=test_scores,
    )
