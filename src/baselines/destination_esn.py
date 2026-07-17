"""Deterministic destination-task echo-state reservoir features."""

from __future__ import annotations

import numpy as np


def make_reservoir(n_inputs: int, n_units: int, spectral_radius: float, input_scale: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w_in = rng.normal(0.0, input_scale, size=(n_units, n_inputs))
    w = rng.normal(0.0, 1.0 / np.sqrt(n_units), size=(n_units, n_units))
    radius = float(np.max(np.abs(np.linalg.eigvals(w))))
    w *= spectral_radius / max(radius, 1e-8)
    bias = rng.normal(0.0, 0.1, size=n_units)
    return w_in, w, bias


def reservoir_features(paths: np.ndarray, w_in: np.ndarray, w: np.ndarray, bias: np.ndarray, leak: float) -> np.ndarray:
    rows = []
    for path in paths:
        state = np.zeros(w.shape[0], dtype=float)
        states = []
        for value in path:
            proposal = np.tanh(w_in @ value + w @ state + bias)
            state = (1.0 - leak) * state + leak * proposal
            states.append(state.copy())
        states = np.asarray(states)
        rows.append(np.r_[states[-1], states.mean(axis=0)])
    return np.asarray(rows)
