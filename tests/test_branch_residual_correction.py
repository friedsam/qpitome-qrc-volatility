from __future__ import annotations

import numpy as np

from qpitome_qrc.baselines.branch_residual_correction import (
    OffsetCorrectionConfig,
    fit_offset_logit_correction,
    fit_residual_pls,
    logit_from_probability,
)


def test_zero_scores_preserve_baseline_probability() -> None:
    p_train = np.asarray([0.2, 0.8, 0.3, 0.7], dtype=float)
    y_train = np.asarray([0, 1, 0, 1], dtype=int)
    train_scores = np.zeros((4, 1), dtype=float)
    test_scores = np.zeros((1, 1), dtype=float)
    p_test = 0.63

    corrected, beta = fit_offset_logit_correction(
        train_scores,
        y_train,
        logit_from_probability(p_train),
        test_scores,
        float(logit_from_probability([p_test])[0]),
    )

    assert np.allclose(beta, 0.0)
    assert np.isclose(corrected, p_test)


def test_offset_correction_moves_probability_in_learned_direction() -> None:
    scores = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    y_train = np.asarray([0, 0, 1, 1], dtype=int)
    baseline_probability = np.full(4, 0.5, dtype=float)

    corrected, beta = fit_offset_logit_correction(
        scores,
        y_train,
        logit_from_probability(baseline_probability),
        np.asarray([[1.5]], dtype=float),
        0.0,
        OffsetCorrectionConfig(l2_penalty=0.1),
    )

    assert beta[0] > 0.0
    assert corrected > 0.5


def test_residual_pls_returns_requested_train_only_dimension() -> None:
    rng = np.random.default_rng(7)
    train_states = rng.normal(size=(12, 8))
    residual = 0.7 * train_states[:, 0] - 0.2 * train_states[:, 1]
    test_state = rng.normal(size=8)

    train_scores, test_scores = fit_residual_pls(
        train_states,
        residual,
        test_state,
        n_components=2,
    )

    assert train_scores.shape == (12, 2)
    assert test_scores.shape == (1, 2)
    assert np.isfinite(train_scores).all()
    assert np.isfinite(test_scores).all()
