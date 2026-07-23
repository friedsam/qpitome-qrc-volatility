from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.l5_template_increment_assay import (
    L5TemplateIncrementConfig,
    _fit_centered_no_intercept,
    fit_template_increment_models,
)


def test_config_allows_negative_zero_and_positive_lambda() -> None:
    config = L5TemplateIncrementConfig()
    config.validate()
    assert any(value < 0.0 for value in config.signed_lambda_grid)
    assert 0.0 in config.signed_lambda_grid
    assert any(value > 0.0 for value in config.signed_lambda_grid)


def test_centered_no_intercept_separates_template_from_deviation() -> None:
    x = np.linspace(-1.0, 1.0, 20)[:, None]
    matrix = np.column_stack([x, x**2])
    targets = np.column_stack(
        [
            0.4 + 0.2 * x[:, 0],
            -0.3 - 0.1 * x[:, 0],
        ]
    )
    fit = np.zeros(20, dtype=bool)
    fit[:15] = True
    template, deviation = _fit_centered_no_intercept(
        matrix,
        targets,
        fit_mask=fit,
        alpha=1.0,
    )
    assert template.shape == (2,)
    assert deviation.shape == targets.shape
    assert np.allclose(template, targets[fit].mean(axis=0))
    assert np.isfinite(deviation).all()


def test_template_increment_models_are_complete() -> None:
    rng = np.random.default_rng(7)
    rows = 18
    matrix = rng.normal(size=(rows, 9))
    residuals = np.zeros((rows, 10), dtype=float)
    residuals[:, :4] = -0.15 + 0.08 * matrix[:, [0]]
    residuals[:, 4:] = 0.35 - 0.12 * matrix[:, [1]]
    har = np.zeros_like(residuals)
    y = residuals.copy()
    specialist_train = np.zeros(rows, dtype=bool)
    specialist_train[:12] = True
    origin_date = (
        pd.date_range("2022-01-01", periods=rows, freq="D", tz="UTC")
        .astype(str)
        .to_numpy()
    )
    config = L5TemplateIncrementConfig(
        minimum_specialist_rows=6,
        minimum_cv_fit_rows=4,
        alpha_grid=(1.0, 10.0),
        signed_lambda_grid=(-1.0, 0.0, 1.0),
    )
    corrections, diagnostics, candidates = fit_template_increment_models(
        matrix,
        residuals=residuals,
        har=har,
        y=y,
        specialist_train=specialist_train,
        origin_date=origin_date,
        config=config,
    )
    assert set(corrections) == {
        "crisis_template_only",
        "template_plus_qrc_unit",
        "template_plus_qrc_signed",
    }
    for correction in corrections.values():
        assert correction.shape == residuals.shape
        assert np.isfinite(correction).all()
    assert diagnostics["lambda_early"] in config.signed_lambda_grid
    assert diagnostics["lambda_late"] in config.signed_lambda_grid
    assert set(candidates["head"]) == {"early", "late"}
    template = corrections["crisis_template_only"]
    assert np.allclose(template, template[[0]])
