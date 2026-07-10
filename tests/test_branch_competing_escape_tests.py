from __future__ import annotations

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_competing_escape_tests import (
    CompetingEscapeTestConfig,
    summarize_clock_comparison,
    summarize_direction_predictions,
    summarize_hazard_predictions,
)


def test_direction_summary_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "landmark_day": [5, 5, 5, 5],
            "model": ["D1_geometry"] * 4,
            "y_recovery": [0, 0, 1, 1],
            "p_recovery": [0.1, 0.2, 0.8, 0.9],
        }
    )
    summary = summarize_direction_predictions(predictions)
    assert summary.iloc[0]["n_oos"] == 4
    assert summary.iloc[0]["auc"] == 1.0
    assert summary.iloc[0]["logloss"] < 0.3


def test_hazard_summary_prefers_lower_episode_nll() -> None:
    scores = pd.DataFrame(
        {
            "episode_id": [1, 2, 1, 2],
            "model": ["H0", "H0", "H1", "H1"],
            "episode_nll": [3.0, 3.0, 2.0, 2.0],
            "episode_loglik": [-3.0, -3.0, -2.0, -2.0],
        }
    )
    summary = summarize_hazard_predictions(scores)
    assert summary.iloc[0]["model"] == "H1"
    assert np.isclose(summary.iloc[0]["mean_episode_nll"], 2.0)


def test_clock_summary_prefers_higher_loglik() -> None:
    scores = pd.DataFrame(
        {
            "episode_id": [1, 2, 1, 2],
            "model": ["T0_exponential", "T0_exponential", "T1_weibull", "T1_weibull"],
            "loglik": [-4.0, -4.0, -3.0, -3.0],
            "shape": [1.0, 1.0, 1.2, 1.1],
        }
    )
    summary = summarize_clock_comparison(scores)
    assert summary.iloc[0]["model"] == "T1_weibull"
    assert np.isclose(summary.iloc[0]["mean_nll"], 3.0)


def test_default_landmarks_are_early_and_fixed() -> None:
    cfg = CompetingEscapeTestConfig()
    assert cfg.landmarks == (5, 10, 15)
