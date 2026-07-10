from __future__ import annotations

import numpy as np

from qpitome_qrc.regimes.branch_relaxation_dynamics import (
    lag1_autocorrelation,
    restoring_fraction,
    same_side_run_length,
    shock_carryover,
)


def test_lag1_autocorrelation_detects_persistence() -> None:
    x = np.asarray([0.0, 1.0, 1.8, 2.5, 3.1, 3.6, 4.0, 4.3])
    assert np.isfinite(lag1_autocorrelation(x))


def test_restoring_fraction_is_high_for_alternating_return_to_center() -> None:
    x = np.asarray([2.0, 0.0, -2.0, 0.0, 2.0, 0.0, -2.0])
    assert restoring_fraction(x) > 0.5


def test_same_side_run_length_detects_persistent_excursions() -> None:
    x = np.asarray([3.0, 3.0, 3.0, -1.0, -1.0, -1.0])
    assert same_side_run_length(x) >= 2.0


def test_shock_carryover_reports_persistent_response() -> None:
    ret = np.asarray([0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0], dtype=float)
    response = np.asarray([0, 0, 0, 0, 0, 1, 2, 1.8, 1.6, 1.4, 1.2, 1.0], dtype=float)
    carry = shock_carryover(ret, response, baseline_window=5, lags=(1, 3, 5))
    assert np.isclose(carry[1], 1.0)
    assert carry[5] > 0.0
