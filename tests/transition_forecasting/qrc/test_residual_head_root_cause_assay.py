from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.residual_head_root_cause_assay import (
    ResidualHeadRootCauseConfig,
    _fit_readouts,
    _select_lambdas,
    balanced_sign_accuracy,
)


def test_intercept_readout_collapses_to_positive_mean_but_no_intercept_does_not() -> None:
    rng = np.random.default_rng(20260724)
    rows = 400
    matrix = rng.normal(size=(rows, 6))
    residuals = rng.normal(loc=0.25, scale=0.35, size=(rows, 10))
    fit = np.ones(rows, dtype=bool)

    predictions, intercept = _fit_readouts(
        matrix,
        residuals,
        fit,
        alpha=1_000_000.0,
    )

    np.testing.assert_allclose(intercept, residuals.mean(axis=0), atol=1e-10)
    assert (predictions["current_intercept"] > 0.0).mean() > 0.99
    assert np.max(np.abs(predictions["no_intercept"])) < 1e-3
    assert 0.45 <= balanced_sign_accuracy(
        residuals, predictions["current_intercept"]
    ) <= 0.55


def test_early_lambda_selection_is_not_driven_by_late_horizons() -> None:
    rows = 80
    horizons = 10
    config = ResidualHeadRootCauseConfig()
    y = np.zeros((rows, horizons), dtype=float)
    har = np.zeros_like(y)
    correction = np.zeros_like(y)
    correction[:, :4] = 1.0
    y[:, 4:] = 2.0
    correction[:, 4:] = 2.0
    tune = np.ones(rows, dtype=bool)

    early, late, _ = _select_lambdas(y, har, correction, tune, config)

    assert early == 0.0
    assert late == 1.0


def test_positive_early_lambda_requires_early_panel_benefit() -> None:
    rows = 80
    horizons = 10
    config = ResidualHeadRootCauseConfig()
    y = np.zeros((rows, horizons), dtype=float)
    har = np.zeros_like(y)
    correction = np.zeros_like(y)
    y[:, :4] = 0.5
    correction[:, :4] = 1.0
    tune = np.ones(rows, dtype=bool)

    early, late, _ = _select_lambdas(y, har, correction, tune, config)

    assert early == 0.5
    assert late >= early
