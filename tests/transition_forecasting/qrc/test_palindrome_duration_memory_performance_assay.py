from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.palindrome_duration_memory_performance_assay import (
    PalindromeDurationMemoryPerformanceConfig,
    fit_duration_selected_residual_model,
    fit_real_history_memory,
)


def _frame(rows: int) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "origin_date": dates.astype(str),
            "sample_id": [f"sample_{index}" for index in range(rows)],
            "label": np.tile([0, 1], rows // 2),
            "lead": np.full(rows, 5),
        }
    )


def test_default_duration_configuration_is_bounded_and_leakage_safe() -> None:
    config = PalindromeDurationMemoryPerformanceConfig()
    config.validate()
    assert config.max_per_class == 24
    assert config.durations_us == (0.010, 0.015, 0.020, 0.025, 0.030)
    assert config.schedule_names == ("crossover_Ahalf_B_Ahalf",)
    assert set(config.interaction_labels) == {"on", "off"}
    assert 0.0 in config.correction_lambdas


def test_duration_configuration_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="durations_us"):
        PalindromeDurationMemoryPerformanceConfig(durations_us=(0.02, 0.02)).validate()
    with pytest.raises(ValueError, match="outside"):
        PalindromeDurationMemoryPerformanceConfig(memory_delays=(40,)).validate()


def test_joint_duration_selector_is_no_intercept_and_uses_inner_time() -> None:
    rng = np.random.default_rng(20260725)
    rows = 80
    horizons = len(TARGET_COLUMNS)
    frame = _frame(rows)
    latent = rng.normal(size=(rows, 4))
    coefficient = rng.normal(scale=0.04, size=(4, horizons))
    har = rng.normal(scale=0.1, size=(rows, horizons))
    residuals = latent @ coefficient + rng.normal(scale=0.01, size=(rows, horizons))
    y = har + residuals
    matrices = {
        0.01: rng.normal(size=(rows, 6)),
        0.02: np.concatenate([latent, rng.normal(scale=0.01, size=(rows, 2))], axis=1),
        0.03: rng.normal(size=(rows, 6)),
    }
    residual_train = np.zeros(rows, dtype=bool)
    residual_train[:60] = True
    validation = ~residual_train
    config = PalindromeDurationMemoryPerformanceConfig(
        folds=(4,),
        durations_us=(0.01, 0.02, 0.03),
        ridge_alphas=(0.1, 1.0),
        correction_lambdas=(0.0, 0.5, 1.0),
        synthetic_samples=100,
        synthetic_seeds=(1,),
    )

    prediction, correction, selection, candidates, fixed = (
        fit_duration_selected_residual_model(
            matrices,
            frame=frame,
            y=y,
            har=har,
            residuals=residuals,
            residual_train_mask=residual_train,
            validation_mask=validation,
            config=config,
        )
    )
    assert prediction.shape == y.shape
    assert correction.shape == y.shape
    assert selection["fit_intercept"] is False
    assert selection["selected_duration_us"] in config.durations_us
    assert len(candidates) == (
        len(config.durations_us)
        * len(config.ridge_alphas)
        * len(config.correction_lambdas)
    )
    assert set(fixed) == set(config.durations_us)


def test_real_history_memory_reports_increment_over_endpoint() -> None:
    rng = np.random.default_rng(31)
    rows = 80
    steps = 12
    frame = _frame(rows)
    encoded = rng.normal(size=(rows, steps, 2))
    # QRC features explicitly contain lag-2 channel 1, so the QRC readout should
    # be able to add information beyond the current endpoint.
    qrc = np.column_stack(
        [
            encoded[:, -3, 0],
            encoded[:, -2, 1],
            rng.normal(scale=0.05, size=rows),
        ]
    )
    train = np.zeros(rows, dtype=bool)
    train[:60] = True
    validation = ~train
    result = fit_real_history_memory(
        qrc,
        encoded,
        frame=frame,
        train_mask=train,
        validation_mask=validation,
        memory_delays=(1, 2),
        alphas=(0.01, 0.1, 1.0),
        inner_holdout_fraction=0.25,
    )
    target = result.loc[
        result["channel"].eq(1)
        & result["delay"].eq(2)
        & result["readout"].eq("qrc")
    ].iloc[0]
    assert target["status"] == "ok"
    assert target["incremental_corr2_vs_endpoint"] > 0.5
