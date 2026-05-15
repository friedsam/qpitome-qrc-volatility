from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    precision_class_1: float
    recall_class_1: float
    f1_class_1: float
    confusion_matrix: np.ndarray


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Evaluate binary classifier with stress class = 1."""
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