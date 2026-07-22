from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.ladder_readout_upgrade_tools import (
    LadderReadoutUpgradeConfig,
    apply_calibration,
    fit_residual_correction,
    select_inner_configuration,
)


def test_segmented_calibration_applies_distinct_horizon_weights() -> None:
    har = np.zeros((2, 10), dtype=float)
    correction = np.ones((2, 10), dtype=float)
    prediction = apply_calibration(
        har,
        correction,
        early_lambda=0.25,
        transition_lambda=1.0,
        split_horizon=4,
    )
    assert np.allclose(prediction[:, :4], 0.25)
    assert np.allclose(prediction[:, 4:], 1.0)


def test_degree_two_readout_has_expected_width_and_finite_output() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(30, 9))
    residuals = rng.normal(size=(30, 10))
    fit = np.zeros(30, dtype=bool)
    fit[:20] = True
    prediction, width = fit_residual_correction(
        matrix,
        residuals,
        fit,
        readout_kind="polynomial_degree2",
        ridge_alpha=100.0,
        polynomial_degree=2,
    )
    assert prediction.shape == residuals.shape
    assert width == 54
    assert np.isfinite(prediction).all()


def test_inner_selection_respects_transition_not_below_early() -> None:
    rng = np.random.default_rng(17)
    rows = 48
    matrix = rng.normal(size=(rows, 9))
    residuals = rng.normal(scale=0.1, size=(rows, 10))
    har = rng.normal(loc=-4.5, scale=0.2, size=(rows, 10))
    y = har + residuals
    residual_train = np.ones(rows, dtype=bool)
    origin_date = np.asarray(
        [f"2020-01-{1 + index // 2:02d}" for index in range(rows)],
        dtype=str,
    )
    config = LadderReadoutUpgradeConfig(
        folds=(1,),
        early_lambdas=(0.0, 0.25, 0.5),
        transition_lambdas=(0.0, 0.25, 0.5, 0.75),
        polynomial_alphas=(100.0,),
    )
    selected, candidates, correction, width = select_inner_configuration(
        matrix,
        y=y,
        har=har,
        residuals=residuals,
        residual_train_mask=residual_train,
        origin_date=origin_date,
        readout_kind="linear",
        calibration_kind="segmented",
        config=config,
    )
    assert selected["transition_lambda"] >= selected["early_lambda"]
    assert (candidates["transition_lambda"] >= candidates["early_lambda"]).all()
    assert correction.shape == y.shape
    assert width == 9
