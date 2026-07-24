from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.ladder_readout_upgrade_tools import (
    fit_residual_correction,
)


def _balanced_sign_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=float).reshape(-1)
    p = np.asarray(prediction, dtype=float).reshape(-1)
    positive = y > 0.0
    negative = y < 0.0
    return 0.5 * ((p[positive] > 0.0).mean() + (p[negative] < 0.0).mean())


def test_no_signal_positive_mean_target_becomes_upward_default() -> None:
    rng = np.random.default_rng(20260723)
    rows = 500
    features = rng.normal(size=(rows, 9))
    residuals = rng.normal(loc=0.20, scale=0.45, size=(rows, 10))
    fit_mask = np.ones(rows, dtype=bool)

    correction, width = fit_residual_correction(
        features,
        residuals,
        fit_mask,
        readout_kind="linear",
        ridge_alpha=1_000_000.0,
    )

    assert width == 9
    # With standardized uninformative features and a very strong penalty, Ridge
    # collapses toward its unpenalized per-horizon intercept: the mean residual.
    np.testing.assert_allclose(
        correction.mean(axis=0), residuals.mean(axis=0), atol=1e-10, rtol=0.0
    )
    assert (correction > 0.0).mean() > 0.99
    assert 0.45 <= _balanced_sign_accuracy(residuals, correction) <= 0.55


def test_informative_signed_feature_recovers_both_residual_directions() -> None:
    rng = np.random.default_rng(9)
    rows = 500
    signal = rng.normal(size=rows)
    features = np.column_stack(
        [signal, rng.normal(scale=0.2, size=(rows, 8))]
    )
    horizon_scale = np.linspace(0.4, 1.2, 10)
    residuals = signal[:, None] * horizon_scale[None, :]
    residuals += rng.normal(scale=0.03, size=residuals.shape)
    fit_mask = np.ones(rows, dtype=bool)

    correction, _ = fit_residual_correction(
        features,
        residuals,
        fit_mask,
        readout_kind="linear",
        ridge_alpha=1.0,
    )

    assert _balanced_sign_accuracy(residuals, correction) > 0.95
    assert np.corrcoef(residuals.reshape(-1), correction.reshape(-1))[0, 1] > 0.98


def test_signed_lambda_only_repairs_a_global_inversion() -> None:
    rng = np.random.default_rng(17)
    residual = rng.normal(size=1000)
    globally_inverted = -residual
    upward_default = np.full_like(residual, 0.25)

    inverted_fixed = -1.0 * globally_inverted
    default_flipped = -1.0 * upward_default

    assert _balanced_sign_accuracy(residual, inverted_fixed) == 1.0
    assert _balanced_sign_accuracy(residual, default_flipped) == 0.5
