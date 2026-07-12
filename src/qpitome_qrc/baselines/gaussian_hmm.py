"""Small Gaussian hidden-Markov models for regime-switching baselines.

The implementation is intentionally narrow: one scalar observation per time
step, state-specific means and variances, optional transition masks, and EM
estimation with multiple deterministic initializations.  It is designed for
transparent classical baselines rather than as a general HMM package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass
class GaussianHMMFit:
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool


def maheu_restricted_mask() -> np.ndarray:
    """Return the four-state transition mask described by Maheu et al.

    State order:
      0 bear, 1 bear rally, 2 bull correction, 3 bull.
    """
    return np.array(
        [
            [1, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
        ],
        dtype=bool,
    )


def _normal_pdf(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    variances = np.maximum(np.asarray(variances, dtype=float), _EPS)
    z = x[:, None] - means[None, :]
    return np.exp(-0.5 * z * z / variances[None, :]) / np.sqrt(
        2.0 * np.pi * variances[None, :]
    )


def _normalize_rows(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.where(mask, np.maximum(matrix, _EPS), 0.0)
    sums = out.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Each transition row must allow at least one state")
    return out / sums


def forward_filter(
    x: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward filter returning filtered probabilities and scales."""
    x = np.asarray(x, dtype=float).reshape(-1)
    emission = np.maximum(_normal_pdf(x, means, variances), _EPS)
    filtered = np.empty_like(emission)
    scales = np.empty(len(x), dtype=float)

    alpha = np.asarray(initial, dtype=float) * emission[0]
    scales[0] = max(float(alpha.sum()), _EPS)
    filtered[0] = alpha / scales[0]
    for t in range(1, len(x)):
        alpha = (filtered[t - 1] @ transition) * emission[t]
        scales[t] = max(float(alpha.sum()), _EPS)
        filtered[t] = alpha / scales[t]
    return filtered, scales, float(np.log(scales).sum())


def forward_backward(
    x: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return posterior state and transition probabilities for EM."""
    x = np.asarray(x, dtype=float).reshape(-1)
    emission = np.maximum(_normal_pdf(x, means, variances), _EPS)
    filtered, scales, log_likelihood = forward_filter(
        x, initial, transition, means, variances
    )
    beta = np.ones_like(filtered)
    for t in range(len(x) - 2, -1, -1):
        beta[t] = transition @ (emission[t + 1] * beta[t + 1])
        beta[t] /= max(scales[t + 1], _EPS)

    gamma = filtered * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _EPS)

    xi = np.empty((max(0, len(x) - 1), len(initial), len(initial)), dtype=float)
    for t in range(len(x) - 1):
        numer = (
            filtered[t][:, None]
            * transition
            * (emission[t + 1] * beta[t + 1])[None, :]
        )
        xi[t] = numer / max(float(numer.sum()), _EPS)
    return gamma, xi, log_likelihood


def _initial_parameters(
    x: np.ndarray,
    n_states: int,
    mask: np.ndarray,
    seed: int,
    mean_signs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0.1, 0.9, n_states)
    means = np.quantile(x, quantiles)
    means += rng.normal(scale=max(float(np.std(x)), 1e-3) * 0.05, size=n_states)
    if mean_signs is not None:
        scale = max(float(np.std(x)), 1e-3)
        means = np.where(mean_signs < 0, -np.maximum(np.abs(means), 0.05 * scale), means)
        means = np.where(mean_signs > 0, np.maximum(np.abs(means), 0.05 * scale), means)
    base_var = max(float(np.var(x)), 1e-4)
    variances = base_var * np.exp(rng.normal(scale=0.4, size=n_states))
    transition = np.where(mask, 1.0, 0.0)
    transition += np.eye(n_states) * 8.0
    transition += np.where(mask, rng.uniform(0.0, 0.2, size=(n_states, n_states)), 0.0)
    transition = _normalize_rows(transition, mask)
    initial = np.full(n_states, 1.0 / n_states)
    return initial, transition, means, variances


def fit_gaussian_hmm(
    x: np.ndarray,
    n_states: int,
    transition_mask: np.ndarray | None = None,
    mean_signs: np.ndarray | None = None,
    n_starts: int = 8,
    max_iter: int = 500,
    tol: float = 1e-7,
    variance_floor: float = 1e-4,
    random_seed: int = 20260711,
) -> GaussianHMMFit:
    """Fit a scalar Gaussian HMM by masked Baum-Welch EM."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < max(20, 5 * n_states):
        raise ValueError("Too few observations for requested HMM")
    mask = (
        np.ones((n_states, n_states), dtype=bool)
        if transition_mask is None
        else np.asarray(transition_mask, dtype=bool)
    )
    if mask.shape != (n_states, n_states):
        raise ValueError("transition_mask has wrong shape")
    signs = None if mean_signs is None else np.asarray(mean_signs, dtype=int)
    if signs is not None and signs.shape != (n_states,):
        raise ValueError("mean_signs has wrong shape")

    best: GaussianHMMFit | None = None
    for start in range(n_starts):
        initial, transition, means, variances = _initial_parameters(
            x, n_states, mask, random_seed + start, signs
        )
        previous = -np.inf
        converged = False
        for iteration in range(1, max_iter + 1):
            gamma, xi, ll = forward_backward(
                x, initial, transition, means, variances
            )
            weights = np.maximum(gamma.sum(axis=0), _EPS)
            initial = np.maximum(gamma[0], _EPS)
            initial /= initial.sum()
            if len(xi):
                transition = _normalize_rows(xi.sum(axis=0), mask)
            means = (gamma * x[:, None]).sum(axis=0) / weights
            if signs is not None:
                min_abs = max(float(np.std(x)) * 1e-4, 1e-8)
                means = np.where(signs < 0, -np.maximum(np.abs(means), min_abs), means)
                means = np.where(signs > 0, np.maximum(np.abs(means), min_abs), means)
            residual = x[:, None] - means[None, :]
            variances = np.maximum(
                (gamma * residual * residual).sum(axis=0) / weights,
                variance_floor,
            )
            if np.isfinite(previous) and abs(ll - previous) <= tol * (1.0 + abs(previous)):
                converged = True
                break
            previous = ll

        _, _, final_ll = forward_filter(x, initial, transition, means, variances)
        fit = GaussianHMMFit(
            initial=initial.copy(),
            transition=transition.copy(),
            means=means.copy(),
            variances=variances.copy(),
            log_likelihood=float(final_ll),
            n_iter=iteration,
            converged=converged,
        )
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit
    assert best is not None
    return best


def one_step_predictive_density(
    observation: float,
    previous_filtered: np.ndarray,
    fit: GaussianHMMFit,
) -> tuple[float, np.ndarray]:
    """Evaluate one-step density and update filtered state probabilities."""
    predictive_state = np.asarray(previous_filtered, dtype=float) @ fit.transition
    emission = _normal_pdf(np.array([observation]), fit.means, fit.variances)[0]
    joint = predictive_state * emission
    density = max(float(joint.sum()), _EPS)
    return density, joint / density
