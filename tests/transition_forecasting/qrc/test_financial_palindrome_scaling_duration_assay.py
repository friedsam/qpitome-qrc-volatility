from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.financial_palindrome_scaling_duration_assay import (
    FinancialPalindromeScalingDurationConfig,
    _aggregate,
    _candidate_name,
    _encoding_diagnostics,
)
from transition_forecasting.qrc.representation_candidates import (
    fit_channel_scaler,
    transform_candidate_sequences,
)


def test_config_rejects_invalid_quantiles() -> None:
    with pytest.raises(ValueError, match="invalid scaling quantiles"):
        FinancialPalindromeScalingDurationConfig(
            scaling_quantiles=((0.9, 0.1),)
        ).validate()


def test_candidate_name_is_stable_and_filesystem_safe() -> None:
    name = _candidate_name(0.01, 0.99, 0.02, "on")
    assert name == "q0p010_0p990__dt0p020__on"
    assert "." not in name


def test_fold_local_scaling_changes_clipping_without_validation_leakage() -> None:
    sequences = np.zeros((8, 4, 2), dtype=float)
    sequences[:4, :, 0] = np.linspace(-1.0, 1.0, 16).reshape(4, 4)
    sequences[:4, :, 1] = np.linspace(-2.0, 2.0, 16).reshape(4, 4)
    sequences[4:, :, :] = 100.0
    train = np.array([True, True, True, True, False, False, False, False])
    validation = ~train
    scaler = fit_channel_scaler(sequences, train, q_low=0.05, q_high=0.95)
    encoded = transform_candidate_sequences(sequences, scaler)
    diagnostics = _encoding_diagnostics(encoded, train, validation)
    assert scaler.medians[0] == pytest.approx(0.0)
    assert diagnostics["channel0_validation_clip_fraction"] == pytest.approx(1.0)
    assert diagnostics["channel0_train_clip_fraction"] < 1.0


def test_aggregate_prioritizes_stable_direction_before_mean_loss() -> None:
    rows = []
    for fold in (4, 5, 6):
        rows.append(
            {
                "fold": fold,
                "q_low": 0.01,
                "q_high": 0.99,
                "duration_us": 0.02,
                "interaction": "on",
                "qlike_delta": -0.01,
                "rmse_delta": -0.01,
                "correction_residual_correlation": 0.10,
                "residual_sign_gap": 0.05,
                "balanced_sign_accuracy": 0.55,
                "wrong_up_rate": 0.40,
                "wrong_down_rate": 0.40,
                "mean_abs_correction": 0.02,
            }
        )
        rows.append(
            {
                "fold": fold,
                "q_low": 0.05,
                "q_high": 0.95,
                "duration_us": 0.015,
                "interaction": "on",
                "qlike_delta": -0.10,
                "rmse_delta": -0.10,
                "correction_residual_correlation": -0.10,
                "residual_sign_gap": -0.05,
                "balanced_sign_accuracy": 0.45,
                "wrong_up_rate": 0.60,
                "wrong_down_rate": 0.60,
                "mean_abs_correction": 0.10,
            }
        )
    aggregate = _aggregate(pd.DataFrame(rows))
    first = aggregate.iloc[0]
    assert first["q_low"] == pytest.approx(0.01)
    assert first["duration_us"] == pytest.approx(0.02)
    assert first["positive_correlation_folds"] == 3
