"""Tests for the transparent Gaussian HMM baseline."""

from __future__ import annotations

import numpy as np

from qpitome_qrc.baselines.gaussian_hmm import (
    fit_gaussian_hmm,
    forward_filter,
    maheu_restricted_mask,
    one_step_predictive_density,
)


def test_maheu_mask_has_expected_zeros() -> None:
    mask = maheu_restricted_mask()
    assert mask.shape == (4, 4)
    assert not mask[0, 2]
    assert not mask[1, 2]
    assert not mask[2, 1]
    assert not mask[3, 1]
    assert mask[0, 3]
    assert mask[3, 0]


def test_forward_filter_probabilities_normalize() -> None:
    x = np.array([-1.0, -0.5, 0.3, 0.9])
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    means = np.array([-0.5, 0.5])
    variances = np.array([0.4, 0.4])
    filtered, scales, ll = forward_filter(x, initial, transition, means, variances)
    assert np.allclose(filtered.sum(axis=1), 1.0)
    assert np.all(scales > 0)
    assert np.isfinite(ll)


def test_fit_two_state_separated_series() -> None:
    rng = np.random.default_rng(5)
    x = np.concatenate([
        rng.normal(-1.0, 0.3, 80),
        rng.normal(1.0, 0.3, 80),
    ])
    fit = fit_gaussian_hmm(
        x,
        n_states=2,
        mean_signs=np.array([-1, 1]),
        n_starts=3,
        max_iter=200,
        random_seed=9,
    )
    assert fit.means[0] < 0
    assert fit.means[1] > 0
    assert np.allclose(fit.transition.sum(axis=1), 1.0)
    assert np.isfinite(fit.log_likelihood)


def test_one_step_predictive_update_normalizes() -> None:
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(-0.5, 0.4, 50), rng.normal(0.5, 0.4, 50)])
    fit = fit_gaussian_hmm(
        x,
        n_states=2,
        mean_signs=np.array([-1, 1]),
        n_starts=2,
        max_iter=100,
    )
    filtered, _, _ = forward_filter(x, fit.initial, fit.transition, fit.means, fit.variances)
    density, updated = one_step_predictive_density(0.2, filtered[-1], fit)
    assert density > 0
    assert np.isclose(updated.sum(), 1.0)
