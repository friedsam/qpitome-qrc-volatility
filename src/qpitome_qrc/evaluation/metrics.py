from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


_EPS = 1e-12


@dataclass
class ClassificationMetrics:
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    precision_class_1: float
    recall_class_1: float
    f1_class_1: float
    confusion_matrix: np.ndarray


@dataclass
class VolatilityForecastMetrics:
    rmse: float
    qlike: float
    mz_alpha: float
    mz_beta: float
    mz_r2: float


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Evaluate binary classifier with positive class = 1."""
    y_pred = (y_score >= threshold).astype(int)

    return ClassificationMetrics(
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        roc_auc=roc_auc_score(y_true, y_score),
        pr_auc=average_precision_score(y_true, y_score),
        precision_class_1=precision_score(y_true, y_pred, zero_division=0),
        recall_class_1=recall_score(y_true, y_pred, zero_division=0),
        f1_class_1=f1_score(y_true, y_pred, zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error for volatility or variance forecasts."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = _EPS) -> float:
    """QLIKE loss for positive volatility/variance forecasts.

    This implementation uses the common volatility-forecast form:
        log(pred) + true / pred
    constants independent of the forecast are omitted, so values are for model
    comparison on the same target only.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)

    return float(np.mean(np.log(y_pred) + y_true / y_pred))


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Fit realized = alpha + beta * forecast and return alpha, beta, R^2.

    This is the descriptive Mincer-Zarnowitz regression needed for Phase 2.
    Formal joint hypothesis testing alpha=0 and beta=1 can be added later with
    statsmodels if needed.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    x = y_pred[mask]

    if len(y) < 3:
        raise ValueError("Need at least 3 finite observations for Mincer-Zarnowitz regression.")

    X = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = alpha + beta * x

    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"alpha": float(alpha), "beta": float(beta), "r2": float(r2)}


def evaluate_volatility_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    use_variance_for_qlike: bool = True,
) -> VolatilityForecastMetrics:
    """Evaluate Track A volatility forecasts with RMSE, QLIKE, and MZ.

    If targets are volatility, QLIKE is usually more stable on variance, so by
    default we square true and predicted values before computing QLIKE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    target_for_qlike = y_true**2 if use_variance_for_qlike else y_true
    pred_for_qlike = y_pred**2 if use_variance_for_qlike else y_pred

    mz = mincer_zarnowitz(y_true, y_pred)

    return VolatilityForecastMetrics(
        rmse=rmse(y_true, y_pred),
        qlike=qlike(target_for_qlike, pred_for_qlike),
        mz_alpha=mz["alpha"],
        mz_beta=mz["beta"],
        mz_r2=mz["r2"],
    )


def volatility_metrics_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Convert metric rows into a consistently sorted DataFrame."""
    return pd.DataFrame(rows).sort_values(["target", "rmse"]).reset_index(drop=True)


def print_metrics(name: str, metrics: ClassificationMetrics) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print(f"balanced_accuracy: {metrics.balanced_accuracy:.3f}")
    print(f"roc_auc: {metrics.roc_auc:.3f}")
    print(f"pr_auc: {metrics.pr_auc:.3f}")
    print(f"precision_class_1: {metrics.precision_class_1:.3f}")
    print(f"recall_class_1: {metrics.recall_class_1:.3f}")
    print(f"f1_class_1: {metrics.f1_class_1:.3f}")
    print("confusion_matrix:")
    print(metrics.confusion_matrix)
