"""Unit tests for the reusable GARCH baseline mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from qpitome_qrc.baselines.garch import (
    GARCHConfig,
    variance_path_to_realized_volatility,
)


def test_variance_path_to_realized_volatility_preserves_units() -> None:
    path_pct2 = np.array([1.0, 4.0, 9.0])
    expected = np.sqrt((1.0 + 4.0 + 9.0) / 100.0**2)

    actual = variance_path_to_realized_volatility(
        path_pct2,
        return_scale=100.0,
    )

    assert actual == pytest.approx(expected)


def test_variance_path_nan_propagates() -> None:
    actual = variance_path_to_realized_volatility(
        np.array([1.0, np.nan]),
        return_scale=100.0,
    )

    assert np.isnan(actual)


def test_variance_path_rejects_negative_variance() -> None:
    with pytest.raises(ValueError, match="negative"):
        variance_path_to_realized_volatility(np.array([1.0, -0.1]))


def test_garch_config_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="return_scale"):
        GARCHConfig(return_scale=0.0)
