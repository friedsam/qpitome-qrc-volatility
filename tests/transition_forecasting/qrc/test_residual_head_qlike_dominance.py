from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.modeling.stage_e_classical_baselines import (
    qlike_loss as production_qlike_loss,
)
from transition_forecasting.qrc.residual_head_qlike_dominance import (
    QlikeDominanceConfig,
    annotate_selected_lambdas,
    bounded_cellwise_optimal_lambda,
    lambda_grid_summary,
    qlike_cell_loss,
    qlike_gain_concentration,
    qlike_gradient_wrt_lambda,
)


def test_cell_loss_matches_production_qlike_in_unclipped_range() -> None:
    y = np.asarray([[-5.0, -4.5, -3.8]])
    har = np.asarray([[-4.8, -4.8, -4.1]])
    correction = np.asarray([[0.2, -0.1, 0.4]])
    lambda_value = 0.75

    expected = production_qlike_loss(y, har + lambda_value * correction)
    observed = qlike_cell_loss(y - har, correction, lambda_value)

    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_wrong_sign_correction_has_zero_bounded_qlike_optimum() -> None:
    residual = np.asarray([0.8, -0.8])
    correction = np.asarray([-0.2, 0.2])

    optimum = bounded_cellwise_optimal_lambda(
        residual,
        correction,
        lambda_cap=1.25,
    )

    np.testing.assert_array_equal(optimum, np.zeros(2))
    assert np.all(
        qlike_cell_loss(residual, correction, 0.25)
        > qlike_cell_loss(residual, correction, 0.0)
    )


def test_right_sign_optimum_is_residual_ratio_and_respects_cap() -> None:
    residual = np.asarray([0.4, -0.4, 1.0])
    correction = np.asarray([0.8, -0.2, 0.2])

    optimum = bounded_cellwise_optimal_lambda(
        residual,
        correction,
        lambda_cap=1.25,
    )

    np.testing.assert_allclose(optimum, np.asarray([0.5, 1.25, 1.25]))
    gradient = qlike_gradient_wrt_lambda(
        residual[:1],
        correction[:1],
        0.5,
    )[0]
    assert np.isclose(gradient, 0.0)


def test_qlike_penalizes_equal_underprediction_more_than_overprediction() -> None:
    magnitude = np.asarray([1.0])
    zero = np.asarray([0.0])

    underprediction = qlike_cell_loss(magnitude, zero, 0.0)[0]
    overprediction = qlike_cell_loss(-magnitude, zero, 0.0)[0]

    assert underprediction > 3.0 * overprediction


def test_small_positive_residual_tail_can_select_upward_lambda_against_majority() -> None:
    config = QlikeDominanceConfig()
    residual = np.concatenate(
        [np.full(90, -0.2, dtype=float), np.full(10, 1.0, dtype=float)]
    )
    correction = np.full(100, 0.2, dtype=float)
    cells = pd.DataFrame(
        {
            "fold": 1,
            "model_family": "density_curvature",
            "readout": "current_intercept",
            "population": "validation",
            "segment": "early",
            "har_residual": residual,
            "raw_correction": correction,
        }
    )

    grid = lambda_grid_summary(cells, config)
    assert grid["qlike_selected_lambda"].iat[0] == 0.75
    assert grid["rmse_selected_lambda"].iat[0] == 0.0
    assert grid["wrong_sign_fraction"].iat[0] == 0.9
    assert qlike_gradient_wrt_lambda(residual, correction, 0.0).sum() < 0.0

    selections = pd.DataFrame(
        {
            "fold": [1],
            "model_family": ["density_curvature"],
            "readout": ["current_intercept"],
            "selected_early_lambda": [0.75],
            "selected_transition_lambda": [0.75],
        }
    )
    annotated = annotate_selected_lambdas(cells, selections, config)
    concentration = qlike_gain_concentration(annotated, config)
    top_ten = concentration.loc[
        concentration["top_fraction"].eq(0.10)
    ].iloc[0]

    assert top_ten["share_of_positive_gain"] > 0.99
    negative = annotated.loc[annotated["har_residual"].lt(0.0), "qlike_gain"]
    positive = annotated.loc[annotated["har_residual"].gt(0.0), "qlike_gain"]
    assert negative.lt(0.0).all()
    assert positive.gt(0.0).all()
