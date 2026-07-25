from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.bivariate_crossover_assay import CROSSOVER_SCHEDULES
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    PalindromeRealTaskConfig,
    _fit_no_intercept_residual_model,
    apply_temporal_condition,
    evolve_palindrome_probabilities,
    summarize_prediction_cells,
    validate_har_contract,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def test_default_configuration_is_bounded_and_intercept_safe() -> None:
    config = PalindromeRealTaskConfig()
    config.validate()
    assert config.lead == 5
    assert config.representations == ("level_only", "level_instability")
    assert config.temporal_conditions[0] == "ordered"
    assert 0.0 in config.correction_lambdas


def test_temporal_controls_preserve_endpoint_and_expected_history() -> None:
    values = np.arange(2 * 8 * 2, dtype=float).reshape(2, 8, 2)

    shuffled = apply_temporal_condition(
        values,
        "shuffled_keep_endpoint",
        recent_tail_steps=5,
        seed=7,
    )
    np.testing.assert_array_equal(shuffled[:, -1, :], values[:, -1, :])
    for row in range(len(values)):
        for channel in range(2):
            np.testing.assert_array_equal(
                np.sort(shuffled[row, :-1, channel]),
                np.sort(values[row, :-1, channel]),
            )

    recent = apply_temporal_condition(
        values,
        "recent_tail",
        recent_tail_steps=3,
        seed=7,
    )
    np.testing.assert_array_equal(recent[:, -3:, :], values[:, -3:, :])
    expected_prefix = np.repeat(values[:, -3:-2, :], 5, axis=1)
    np.testing.assert_array_equal(recent[:, :5, :], expected_prefix)

    static = apply_temporal_condition(
        values,
        "static_endpoint",
        recent_tail_steps=3,
        seed=7,
    )
    np.testing.assert_array_equal(
        static,
        np.repeat(values[:, -1:, :], values.shape[1], axis=1),
    )


def test_har_contract_rejects_missing_or_nonfinite_columns() -> None:
    rows = 4
    frame = pd.DataFrame(
        {
            "level": np.linspace(0.1, 0.4, rows),
            "mean5": np.linspace(0.1, 0.4, rows),
            "mean20": np.linspace(0.1, 0.4, rows),
            **{name: np.linspace(0.2, 0.5, rows) for name in TARGET_COLUMNS},
        }
    )
    validate_har_contract(frame)
    with pytest.raises(ValueError, match="missing columns"):
        validate_har_contract(frame.drop(columns=["mean20"]))
    bad = frame.copy()
    bad.loc[0, "level"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_har_contract(bad)


def test_nested_residual_head_is_explicitly_no_intercept() -> None:
    rng = np.random.default_rng(17)
    rows = 80
    horizons = len(TARGET_COLUMNS)
    dates = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "origin_date": dates.astype(str),
            "sample_id": [f"sample_{index}" for index in range(rows)],
            "label": np.tile([0, 1], rows // 2),
            "lead": np.full(rows, 5),
        }
    )
    matrix = rng.normal(size=(rows, 6))
    coefficient = rng.normal(size=(6, horizons)) * 0.04
    har = rng.normal(scale=0.1, size=(rows, horizons))
    residuals = matrix @ coefficient + rng.normal(scale=0.01, size=(rows, horizons))
    y = har + residuals
    residual_train = np.zeros(rows, dtype=bool)
    residual_train[:60] = True
    config = PalindromeRealTaskConfig(
        folds=(4,),
        max_per_class=4,
        ridge_alphas=(0.1, 1.0),
        correction_lambdas=(0.0, 0.5, 1.0),
    )

    prediction, correction, selection, candidates = _fit_no_intercept_residual_model(
        matrix,
        frame=frame,
        y=y,
        har=har,
        residuals=residuals,
        residual_train_mask=residual_train,
        config=config,
    )
    assert prediction.shape == y.shape
    assert correction.shape == y.shape
    assert selection["fit_intercept"] is False
    assert selection["selected_alpha"] in config.ridge_alphas
    assert selection["selected_lambda"] in config.correction_lambdas
    assert len(candidates) == len(config.ridge_alphas) * len(config.correction_lambdas)


def test_prediction_summary_exposes_metric_and_direction_failures() -> None:
    cells = pd.DataFrame(
        {
            "fold": [4, 4, 4, 4],
            "model": ["palindrome_ordered_on"] * 4,
            "representation": ["level_instability"] * 4,
            "condition": ["ordered"] * 4,
            "interactions": ["on"] * 4,
            "sample_id": ["a", "a", "b", "b"],
            "origin_date": ["2020-01-01"] * 2 + ["2020-01-02"] * 2,
            "label": [0, 0, 1, 1],
            "lead": [5] * 4,
            "forecast_horizon": [1, 2, 1, 2],
            "y_true": [0.0, 0.1, 0.5, 0.6],
            "har_prediction": [0.2, 0.2, 0.3, 0.3],
            "prediction": [0.3, 0.1, 0.4, 0.5],
            "har_residual": [-0.2, -0.1, 0.2, 0.3],
            "correction": [0.1, -0.1, 0.1, 0.2],
        }
    )
    summary = summarize_prediction_cells(cells, pooled=True)
    path = summary.loc[summary["scope"].eq("path")].iloc[0]
    assert path["wrong_up_rate"] == pytest.approx(0.5)
    assert path["correct_up_rate"] == pytest.approx(1.0)
    assert np.isfinite(path["qlike_delta_vs_har"])
    assert np.isfinite(path["rmse_delta_vs_har"])


def test_palindrome_interaction_switch_changes_only_physical_interactions() -> None:
    rng = np.random.default_rng(20260725)
    windows = rng.uniform(-0.5, 0.5, size=(4, 8, 2))
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.02,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    schedule = next(
        item for item in CROSSOVER_SCHEDULES if item.name == "crossover_Ahalf_B_Ahalf"
    )
    on, on_meta = evolve_palindrome_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=1.25,
        interactions=True,
        drive_phase_rad=0.0,
    )
    off, off_meta = evolve_palindrome_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=1.25,
        interactions=False,
        drive_phase_rad=0.0,
    )
    assert on.shape == off.shape == (4, 2, 64)
    assert on_meta["interactions_enabled"] is True
    assert off_meta["interactions_enabled"] is False
    assert on_meta["interaction_scale_applied"] == pytest.approx(1.25)
    assert off_meta["interaction_scale_applied"] == pytest.approx(0.0)
    assert not np.allclose(on, off)
