from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.residual_direction_audit import (
    ResidualDirectionAuditConfig,
    augment_prediction_frame,
    best_lambda,
    direction_metric_row,
    synthetic_residual_scenarios,
    synthetic_sanity_table,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["historical"] * 4,
            "selection_seed": [1] * 4,
            "fold": [4] * 4,
            "sample_id": ["a", "a", "b", "b"],
            "episode_id": ["e1", "e1", "e2", "e2"],
            "lead": [5] * 4,
            "label": [0, 0, 1, 1],
            "model_name": ["candidate"] * 4,
            "horizon": [1, 5, 1, 5],
            "y_true": [-0.2, 0.6, -0.1, 0.8],
            "har_pred": [0.0, 0.0, 0.0, 0.0],
            "y_pred": [0.1, 0.4, 0.2, 0.6],
            "early_lambda": [0.5] * 4,
            "transition_lambda": [1.0] * 4,
        }
    )


def test_residual_and_correction_sign_conventions() -> None:
    augmented = augment_prediction_frame(_prediction_frame(), split_horizon=4)
    np.testing.assert_allclose(
        augmented["har_residual"].to_numpy(),
        [-0.2, 0.6, -0.1, 0.8],
    )
    np.testing.assert_allclose(
        augmented["applied_correction"].to_numpy(),
        [0.1, 0.4, 0.2, 0.6],
    )
    assert augmented.loc[0, "wrong_up_when_har_overpredicts"]
    assert augmented.loc[2, "wrong_up_when_har_overpredicts"]
    assert augmented.loc[0, "squared_error_change_vs_har"] > 0.0


def test_raw_correction_is_recovered_from_segment_lambda() -> None:
    augmented = augment_prediction_frame(_prediction_frame(), split_horizon=4)
    np.testing.assert_allclose(
        augmented["raw_correction_inferred"].to_numpy(),
        [0.2, 0.4, 0.4, 0.6],
    )


def test_signed_lambda_repairs_global_inversion() -> None:
    config = ResidualDirectionAuditConfig(synthetic_rows=1000)
    synthetic = synthetic_residual_scenarios(rows=1000, seed=7)
    metrics = synthetic_sanity_table(synthetic, config).set_index("scenario")
    inverted = metrics.loc["globally_inverted_signal"]
    assert -1.15 <= inverted["best_signed_lambda"] <= -0.85
    assert inverted["best_nonnegative_lambda"] == 0.0
    assert inverted["signed_rmse_gain"] > 0.2


def test_signed_lambda_cannot_create_sample_specific_sign_skill() -> None:
    config = ResidualDirectionAuditConfig(synthetic_rows=1000)
    synthetic = synthetic_residual_scenarios(rows=1000, seed=11)
    metrics = synthetic_sanity_table(synthetic, config).set_index("scenario")
    upward = metrics.loc["upward_default_no_sign_skill"]
    assert 0.40 <= upward["raw_balanced_sign_accuracy"] <= 0.60
    assert abs(upward["best_signed_lambda"]) <= 0.20
    assert upward["best_signed_balanced_sign_accuracy"] <= 0.60


def test_direction_metric_detects_upward_default() -> None:
    residual = np.array([-0.5, -0.2, 0.2, 0.5])
    correction = np.array([0.1, 0.1, 0.1, 0.1])
    frame = pd.DataFrame(
        {
            "har_residual": residual,
            "applied_correction": correction,
            "raw_correction_inferred": correction,
            "squared_error_change_vs_har": (residual - correction) ** 2
            - residual**2,
            "absolute_error_change_vs_har": np.abs(residual - correction)
            - np.abs(residual),
            "episode_id": ["a", "b", "c", "d"],
        }
    )
    row = direction_metric_row(frame)
    assert row["mean_correction_when_residual_negative"] > 0.0
    assert row["mean_correction_when_residual_positive"] > 0.0
    assert row["positive_correction_rate_when_residual_negative"] == 1.0
    assert row["balanced_residual_sign_accuracy"] == 0.5


def test_best_lambda_uses_residual_not_absolute_deviation() -> None:
    residual = np.array([-1.0, -0.5, 0.5, 1.0])
    correction = residual.copy()
    selected = best_lambda(residual, correction, [-1.0, 0.0, 1.0])
    assert selected["lambda"] == 1.0
    assert selected["rmse"] == 0.0
