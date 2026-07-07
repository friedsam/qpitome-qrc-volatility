import numpy as np
import pytest

from qpitome_qrc.evaluation.metrics import (
    evaluate_volatility_forecast,
    mincer_zarnowitz,
    qlike,
    rmse,
)


def test_rmse_matches_manual_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(4.0 / 3.0))


def test_qlike_matches_manual_value_for_positive_inputs():
    y_true = np.array([1.0, 4.0])
    y_pred = np.array([2.0, 2.0])
    expected = np.mean(np.log(y_pred) + y_true / y_pred)

    assert qlike(y_true, y_pred) == pytest.approx(expected)


def test_qlike_protects_near_zero_inputs():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])

    value = qlike(y_true, y_pred)

    assert np.isfinite(value)


def test_mincer_zarnowitz_recovers_linear_relationship():
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = 0.5 + 2.0 * y_pred

    result = mincer_zarnowitz(y_true, y_pred)

    assert result["alpha"] == pytest.approx(0.5)
    assert result["beta"] == pytest.approx(2.0)
    assert result["r2"] == pytest.approx(1.0)


def test_mincer_zarnowitz_requires_three_finite_observations():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        mincer_zarnowitz(y_true, y_pred)


def test_evaluate_volatility_forecast_returns_finite_metrics():
    y_true = np.array([0.10, 0.15, 0.20, 0.30])
    y_pred = np.array([0.11, 0.14, 0.22, 0.28])

    metrics = evaluate_volatility_forecast(y_true, y_pred)

    assert np.isfinite(metrics.rmse)
    assert np.isfinite(metrics.qlike)
    assert np.isfinite(metrics.mz_alpha)
    assert np.isfinite(metrics.mz_beta)
    assert np.isfinite(metrics.mz_r2)
