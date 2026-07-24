from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.residual_alpha_sweep_assay import (
    fit_decomposed_ridge,
)


def test_decomposition_reconstructs_current_ridge_prediction() -> None:
    rng = np.random.default_rng(20260724)
    matrix = rng.normal(size=(120, 6))
    residuals = rng.normal(size=(120, 10))
    fit = np.zeros(120, dtype=bool)
    fit[:80] = True

    result = fit_decomposed_ridge(matrix, residuals, fit, alpha=10.0)

    np.testing.assert_allclose(
        result["total"],
        result["intercept"] + result["feature"],
        atol=1e-12,
        rtol=1e-12,
    )


def test_standardization_makes_intercept_invariant_to_alpha_and_features() -> None:
    rng = np.random.default_rng(31)
    residuals = rng.normal(loc=0.15, scale=0.4, size=(160, 10))
    fit = np.zeros(160, dtype=bool)
    fit[:120] = True
    matrix_a = rng.normal(size=(160, 6))
    matrix_b = np.column_stack(
        [
            np.tanh(2.0 * matrix_a[:, 0]),
            matrix_a[:, 1] * matrix_a[:, 2],
            rng.normal(size=(160, 4)),
        ]
    )

    low = fit_decomposed_ridge(matrix_a, residuals, fit, alpha=0.3)
    high = fit_decomposed_ridge(matrix_a, residuals, fit, alpha=300.0)
    changed_features = fit_decomposed_ridge(matrix_b, residuals, fit, alpha=10.0)

    expected = residuals[fit].mean(axis=0)
    np.testing.assert_allclose(low["intercept_path"], expected, atol=1e-12)
    np.testing.assert_allclose(high["intercept_path"], expected, atol=1e-12)
    np.testing.assert_allclose(changed_features["intercept_path"], expected, atol=1e-12)


def test_larger_alpha_suppresses_feature_dependent_correction() -> None:
    rng = np.random.default_rng(73)
    matrix = rng.normal(size=(300, 6))
    weights = rng.normal(size=(6, 10))
    residuals = matrix @ weights + rng.normal(scale=0.02, size=(300, 10))
    fit = np.ones(300, dtype=bool)

    low = fit_decomposed_ridge(matrix, residuals, fit, alpha=0.1)
    high = fit_decomposed_ridge(matrix, residuals, fit, alpha=300.0)

    assert float(high["coefficient_l2"]) < float(low["coefficient_l2"])
    assert float(high["feature_rms"]) < float(low["feature_rms"])
