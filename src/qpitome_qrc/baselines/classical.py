from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from qpitome_qrc.evaluation.metrics import ClassificationMetrics, evaluate_binary_classifier

ModelKind = Literal[
    "logistic",
    "ridge",
    "linear_svc",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
]


@dataclass(frozen=True)
class ClassicalModelConfig:
    """Configuration for fast tabular classical baselines."""

    name: str
    kind: ModelKind
    scale: bool = True
    class_weight: str | None = "balanced"
    random_state: int = 42
    params: dict[str, Any] | None = None


@dataclass
class ClassicalRunResult:
    config: ClassicalModelConfig
    model: BaseEstimator
    threshold: float
    val_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    val_scores: np.ndarray
    test_scores: np.ndarray
    feature_names: list[str]


def default_model_configs(random_state: int = 42) -> list[ClassicalModelConfig]:
    """Return a compact, useful baseline set.

    No dummy baseline is included by design. These are intended to test whether
    engineered features already contain predictive signal before using ESN/QRC.
    """
    return [
        ClassicalModelConfig(
            name="logistic_l2",
            kind="logistic",
            params={"C": 1.0, "penalty": "l2"},
            random_state=random_state,
        ),
        ClassicalModelConfig(
            name="logistic_l1",
            kind="logistic",
            params={"C": 0.3, "penalty": "l1", "solver": "liblinear"},
            random_state=random_state,
        ),
        ClassicalModelConfig(
            name="ridge_classifier",
            kind="ridge",
            params={"alpha": 1.0},
            random_state=random_state,
        ),
        ClassicalModelConfig(
            name="linear_svc",
            kind="linear_svc",
            params={"C": 1.0},
            random_state=random_state,
        ),
        ClassicalModelConfig(
            name="random_forest",
            kind="random_forest",
            scale=False,
            params={"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 25, "n_jobs": -1},
            random_state=random_state,
        ),
        ClassicalModelConfig(
            name="extra_trees",
            kind="extra_trees",
            scale=False,
            params={"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 25, "n_jobs": -1},
            random_state=random_state,
        ),
        ClassicalModelConfig(
            name="hist_gradient_boosting",
            kind="hist_gradient_boosting",
            scale=False,
            class_weight=None,
            params={"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.1},
            random_state=random_state,
        ),
    ]


def build_model(config: ClassicalModelConfig) -> BaseEstimator:
    """Build sklearn model/pipeline from config."""
    params = dict(config.params or {})

    if config.kind == "logistic":
        model = LogisticRegression(
            class_weight=config.class_weight,
            max_iter=3000,
            random_state=config.random_state,
            **params,
        )
    elif config.kind == "ridge":
        model = RidgeClassifier(class_weight=config.class_weight, random_state=config.random_state, **params)
    elif config.kind == "linear_svc":
        model = LinearSVC(class_weight=config.class_weight, random_state=config.random_state, max_iter=5000, **params)
    elif config.kind == "random_forest":
        model = RandomForestClassifier(class_weight=config.class_weight, random_state=config.random_state, **params)
    elif config.kind == "extra_trees":
        model = ExtraTreesClassifier(class_weight=config.class_weight, random_state=config.random_state, **params)
    elif config.kind == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=config.random_state, **params)
    else:
        raise ValueError(f"Unknown model kind: {config.kind}")

    if config.scale:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def model_scores(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Return continuous positive-class scores for heterogeneous sklearn models."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return np.asarray(raw, dtype=float)
    raise TypeError(f"Model does not expose predict_proba or decision_function: {type(model)}")


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
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def fit_classical_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    config: ClassicalModelConfig,
    tune_threshold: bool = True,
) -> ClassicalRunResult:
    """Fit one classical baseline and evaluate validation/test."""
    model = build_model(config)
    model.fit(X_train, y_train)

    val_scores = model_scores(model, X_val)
    test_scores = model_scores(model, X_test)

    threshold = find_best_threshold(y_val, val_scores)[0] if tune_threshold else 0.5

    val_metrics = evaluate_binary_classifier(y_val, val_scores, threshold=threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_scores, threshold=threshold)

    return ClassicalRunResult(
        config=config,
        model=model,
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_scores=val_scores,
        test_scores=test_scores,
        feature_names=feature_names,
    )


def summarize_classical_result(result: ClassicalRunResult, feature_set: str) -> dict[str, Any]:
    """Flatten one run to a metrics row."""
    row = {
        "feature_set": feature_set,
        "model_name": result.config.name,
        "model_kind": result.config.kind,
        "threshold": result.threshold,
        "n_features": len(result.feature_names),
        "config": asdict(result.config),
        "val_balanced_accuracy": result.val_metrics.balanced_accuracy,
        "val_roc_auc": result.val_metrics.roc_auc,
        "val_pr_auc": result.val_metrics.pr_auc,
        "val_precision_class_1": result.val_metrics.precision_class_1,
        "val_recall_class_1": result.val_metrics.recall_class_1,
        "val_f1_class_1": result.val_metrics.f1_class_1,
        "test_balanced_accuracy": result.test_metrics.balanced_accuracy,
        "test_roc_auc": result.test_metrics.roc_auc,
        "test_pr_auc": result.test_metrics.pr_auc,
        "test_precision_class_1": result.test_metrics.precision_class_1,
        "test_recall_class_1": result.test_metrics.recall_class_1,
        "test_f1_class_1": result.test_metrics.f1_class_1,
    }
    return row


def run_classical_suite(
    arrays_by_feature_set: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    feature_names_by_set: dict[str, list[str]],
    configs: list[ClassicalModelConfig] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], ClassicalRunResult]]:
    """Run model configs over one or more feature sets."""
    if configs is None:
        configs = default_model_configs()

    rows = []
    results: dict[tuple[str, str], ClassicalRunResult] = {}

    for feature_set, arrays in arrays_by_feature_set.items():
        X_train, y_train = arrays["train"]
        X_val, y_val = arrays["val"]
        X_test, y_test = arrays["test"]
        feature_names = feature_names_by_set[feature_set]

        for config in configs:
            print(f"{feature_set} :: {config.name}")
            result = fit_classical_baseline(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                config=config,
                tune_threshold=True,
            )
            results[(feature_set, config.name)] = result
            rows.append(summarize_classical_result(result, feature_set=feature_set))

    summary = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    return summary, results


def permutation_importance_table(
    result: ClassicalRunResult,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = "average_precision",
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance table for any fitted sklearn-compatible model."""
    imp = permutation_importance(
        result.model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": result.feature_names,
            "importance_mean": imp.importances_mean,
            "importance_std": imp.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
