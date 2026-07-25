from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.modeling.l5_residual_foundation import (
    L5ResidualFoundationConfig,
    _qlike_objective_gradient_hessian,
    build_fold_residual_registry,
    exact_qlike_offset,
    fit_mse_har,
    fit_qlike_har,
    qlike_logvol_cells,
    qlike_score_residual,
    selected_residual_paths,
)
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS


def _manifest(rows: int = 180, *, fold: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", periods=rows, freq="7D", tz="UTC")
    x = np.linspace(-1.0, 1.0, rows)
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(rows)],
            "fold": fold,
            "fold_split": np.where(np.arange(rows) < 130, "train", "val"),
            "lead": 5,
            "label": np.arange(rows) % 2,
            "episode_id": [f"e{index}" for index in range(rows)],
            "origin_date": dates.astype(str),
            "level": x,
            "mean5": 0.5 * x,
            "mean20": -0.25 * x,
        }
    )
    rng = np.random.default_rng(7)
    for horizon, column in enumerate(TARGET_COLUMNS, start=1):
        frame[column] = (
            0.1
            + 0.25 * x
            + 0.01 * horizon
            + rng.normal(0.0, 0.08, rows)
        )
    return frame


def test_qlike_offset_is_exact_constant_optimum() -> None:
    residual = np.array([[-0.20], [0.00], [0.15], [0.40]])
    offset = exact_qlike_offset(residual)
    before = qlike_logvol_cells(residual, np.zeros_like(residual)).mean()
    centered = residual - offset[None, :]
    after = qlike_logvol_cells(centered, np.zeros_like(centered)).mean()
    assert after < before
    assert qlike_score_residual(centered).mean(axis=0) == pytest.approx(
        np.zeros(1), abs=1e-12
    )


def test_qlike_har_recovers_variance_shift_and_improves_qlike() -> None:
    rng = np.random.default_rng(11)
    n_train, n_test = 4000, 3000
    x_train = rng.normal(size=(n_train, 3))
    x_test = rng.normal(size=(n_test, 3))
    beta = np.array([0.25, -0.10, 0.05])
    sigma = 0.30
    y_train = (x_train @ beta + rng.normal(0.0, sigma, n_train))[:, None]
    y_test = (x_test @ beta + rng.normal(0.0, sigma, n_test))[:, None]
    mse = fit_mse_har(x_train, y_train, alpha=1e-8)
    qlike = fit_qlike_har(
        x_train,
        y_train,
        alpha=1e-8,
        max_iter=100,
        gradient_tolerance=1e-8,
        initial_model=mse,
    )
    mse_prediction = mse.predict(x_test)
    qlike_prediction = qlike.predict(x_test)
    assert float(np.mean(qlike_prediction - mse_prediction)) == pytest.approx(
        sigma**2, abs=0.02
    )
    assert qlike_logvol_cells(y_test, qlike_prediction).mean() < qlike_logvol_cells(
        y_test, mse_prediction
    ).mean()
    assert np.sqrt(np.mean((y_test - mse_prediction) ** 2)) <= np.sqrt(
        np.mean((y_test - qlike_prediction) ** 2)
    )


def test_qlike_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(19)
    design = np.column_stack([np.ones(30), rng.normal(size=(30, 3))])
    target = rng.normal(0.0, 0.2, 30)
    beta = rng.normal(0.0, 0.1, 4)
    objective, gradient, hessian = _qlike_objective_gradient_hessian(
        beta, design, target, 3.0
    )
    epsilon = 1e-6
    numerical = np.empty_like(beta)
    for index in range(len(beta)):
        direction = np.zeros_like(beta)
        direction[index] = epsilon
        plus, _, _ = _qlike_objective_gradient_hessian(
            beta + direction, design, target, 3.0
        )
        minus, _, _ = _qlike_objective_gradient_hessian(
            beta - direction, design, target, 3.0
        )
        numerical[index] = (plus - minus) / (2.0 * epsilon)
    assert objective > 0.0
    assert np.allclose(gradient, numerical, atol=1e-5)
    assert np.linalg.eigvalsh(hessian).min() > 0.0


def test_registry_uses_target_availability_not_origin_only() -> None:
    frame = _manifest()
    config = L5ResidualFoundationConfig(
        folds=(4,),
        selection_folds=(4,),
        confirmation_folds=(),
        target_availability_calendar_days=21,
    )
    registry, diagnostics = build_fold_residual_registry(
        frame, fold=4, config=config
    )
    first = (
        registry.loc[registry["model"].eq("mse_har")]
        .sort_values("origin_date")
        .iloc[0]
    )
    score_date = pd.Timestamp(first["origin_date"])
    eligible = frame.loc[
        frame["fold_split"].eq("train")
        & (
            pd.to_datetime(frame["origin_date"], utc=True)
            + pd.Timedelta(days=21)
            < score_date
        )
    ]
    assert int(first["fit_rows"]) == len(eligible)
    assert diagnostics["test_rows_used"] == 0


def test_same_date_and_validation_rows_never_enter_fit() -> None:
    frame = _manifest()
    frame.loc[50, "origin_date"] = frame.loc[49, "origin_date"]
    config = L5ResidualFoundationConfig(
        folds=(4,), selection_folds=(4,), confirmation_folds=()
    )
    registry, _ = build_fold_residual_registry(frame, fold=4, config=config)
    mse = registry.loc[registry["model"].eq("mse_har")]
    count_49 = mse.loc[mse["sample_id"].eq("s49"), "fit_rows"].iloc[0]
    count_50 = mse.loc[mse["sample_id"].eq("s50"), "fit_rows"].iloc[0]
    assert count_49 == count_50
    late_validation = (
        mse.loc[mse["fold_split"].eq("val")].sort_values("origin_date").iloc[-1]
    )
    score_date = pd.Timestamp(late_validation["origin_date"])
    eligible = frame.loc[
        frame["fold_split"].eq("train")
        & (
            pd.to_datetime(frame["origin_date"], utc=True)
            + pd.Timedelta(days=21)
            < score_date
        )
    ]
    assert int(late_validation["fit_rows"]) == len(eligible)


def test_registry_never_emits_test_rows() -> None:
    frame = _manifest()
    frame.loc[160:, "fold_split"] = "test"
    config = L5ResidualFoundationConfig(
        folds=(4,), selection_folds=(4,), confirmation_folds=()
    )
    registry, _ = build_fold_residual_registry(frame, fold=4, config=config)
    assert set(registry["fold_split"].unique()).issubset({"train", "val"})


def test_selected_residual_paths_round_trip() -> None:
    config = L5ResidualFoundationConfig(
        folds=(4,), selection_folds=(4,), confirmation_folds=()
    )
    registry, _ = build_fold_residual_registry(
        _manifest(), fold=4, config=config
    )
    keys, truth, prediction, residual = selected_residual_paths(
        registry, model_name="mse_har", target_horizon=10
    )
    assert len(keys) == len(truth) == len(prediction) == len(residual)
    assert np.allclose(truth - prediction, residual)
