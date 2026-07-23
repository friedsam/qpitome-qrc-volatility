from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.modeling.control_strata import CALM, HARD_NEGATIVE, TRANSITION
from transition_forecasting.qrc.transition_signal_readout_assay import (
    TransitionSignalAssayConfig,
    candidate_is_eligible,
    candidate_key,
    transition_calm_mask,
)


def test_transition_calm_scope_excludes_hard_negatives() -> None:
    frame = pd.DataFrame(
        {"evaluation_stratum": [TRANSITION, CALM, HARD_NEGATIVE, TRANSITION]}
    )
    assert np.array_equal(
        transition_calm_mask(frame),
        np.asarray([True, True, False, True]),
    )


def test_candidate_requires_calm_and_l1_guardrails() -> None:
    config = TransitionSignalAssayConfig()
    good = {
        "transition_l5_qlike_gain": 0.10,
        "calm_qlike_delta": 0.01,
        "calm_rmse_delta": 0.01,
        "transition_l1_qlike_delta": 0.03,
    }
    assert candidate_is_eligible(good, config)
    bad = dict(good, calm_qlike_delta=0.03)
    assert not candidate_is_eligible(bad, config)


def test_candidate_key_prioritizes_l5_gain_within_guardrails() -> None:
    base = {
        "eligible": True,
        "transition_all_qlike_gain": 0.05,
        "transition_calm_margin": 0.01,
        "calm_qlike_delta": 0.0,
        "calm_correction_mae": 0.01,
        "fit_intercept": False,
        "ridge_alpha": 100.0,
        "transition_weight": 1.0,
        "early_lambda": 0.25,
        "late_lambda": 1.0,
    }
    weaker = dict(base, transition_l5_qlike_gain=0.02)
    stronger = dict(base, transition_l5_qlike_gain=0.08)
    assert candidate_key(stronger) < candidate_key(weaker)
