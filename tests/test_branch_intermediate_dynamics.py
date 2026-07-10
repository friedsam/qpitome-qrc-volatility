from __future__ import annotations

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    first_passage,
    segment_metrics,
    standardized_mean_difference,
)


def test_first_passage_returns_earliest_barrier() -> None:
    path = np.asarray([0.01, -0.02, -0.08, 0.12])
    day, event = first_passage(path, upper=0.10, lower=-0.07)
    assert day == 3
    assert event == "relapse"


def test_first_passage_can_remain_unresolved() -> None:
    day, event = first_passage(np.asarray([0.01, -0.01, 0.02]), 0.10, -0.10)
    assert day is None
    assert event is None


def test_segment_metrics_detect_switching_and_coherence() -> None:
    segment = pd.DataFrame(
        {
            "signed_return_over_local_vol": [1.0, -1.0, 1.0, -1.0],
            "downside_shock_pressure": [0.1, 0.2, 0.3, 0.4],
            "d_log_rv5_over_rv20": [0.1, -0.2, 0.3, -0.4],
            "drawdown_repair_over_local_vol": [1.0, -1.0, 1.0, -1.0],
        }
    )
    metrics = segment_metrics(segment)
    assert metrics["return_sign_switch_rate"] == 1.0
    assert metrics["repair_sign_switch_rate"] == 1.0
    assert metrics["directional_coherence"] == 0.0
    np.testing.assert_allclose(metrics["rv_ratio_choppiness"], 0.25)
    np.testing.assert_allclose(metrics["mean_downside_pressure"], 0.25)


def test_standardized_mean_difference_has_expected_sign() -> None:
    a = np.asarray([2.0, 3.0, 4.0])
    b = np.asarray([0.0, 1.0, 2.0])
    assert standardized_mean_difference(a, b) > 0
