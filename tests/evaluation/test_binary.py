import numpy as np
import pandas as pd

from evaluation.binary import binary_metric_row, grouped_binary_metrics


def test_binary_metric_row_probability_metrics() -> None:
    row = binary_metric_row(
        "model",
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
    )
    assert row["n_predictions"] == 4
    assert row["n_positive"] == 2
    assert row["n_negative"] == 2
    assert row["roc_auc"] == 1.0
    assert row["accuracy"] == 1.0
    assert row["positive_recall"] == 1.0
    assert row["negative_recall"] == 1.0
    assert 0.0 < row["log_loss"] < 1.0
    assert 0.0 < row["brier"] < 1.0


def test_grouped_binary_metrics_includes_all_and_subgroups() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["m", "m", "m", "m"],
            "y_true": [0, 1, 0, 1],
            "probability_positive": [0.2, 0.8, 0.3, 0.7],
            "origin_regime": ["bear", "bear", "bull", "bull"],
        }
    )
    metrics = grouped_binary_metrics(
        predictions,
        subgroup_col="origin_regime",
        minimum_subgroup_size=2,
    )
    assert set(metrics["subgroup"]) == {"all", "bear", "bull"}


def test_single_class_auc_is_nan() -> None:
    row = binary_metric_row("m", np.array([1, 1]), np.array([0.6, 0.7]))
    assert np.isnan(row["roc_auc"])
    assert np.isnan(row["negative_recall"])
