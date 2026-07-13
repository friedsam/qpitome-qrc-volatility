"""Deterministic echo-state baseline used by the legacy Day-5 experiment."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEFAULT_RESERVOIR_SIZE = 32
DEFAULT_SEED = 23
INPUT_SCALE = 0.45
SPARSITY = 0.18
SPECTRAL_RADIUS = 0.75
BIAS_SCALE = 0.05
LEAK_RETAIN = 0.35
LEAK_UPDATE = 0.65


def fixed_esn_weights(
    input_dim: int,
    n_reservoir: int = DEFAULT_RESERVOIR_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the locked sparse deterministic ESN weights."""

    rng = np.random.default_rng(seed)
    input_weights = rng.normal(0.0, INPUT_SCALE, size=(n_reservoir, input_dim))
    reservoir_weights = rng.normal(0.0, 1.0, size=(n_reservoir, n_reservoir))
    mask = rng.random(reservoir_weights.shape) < SPARSITY
    reservoir_weights *= mask
    eigenvalues = np.linalg.eigvals(reservoir_weights)
    radius = np.max(np.abs(eigenvalues))
    if radius > 0:
        reservoir_weights *= SPECTRAL_RADIUS / radius
    bias = rng.normal(0.0, BIAS_SCALE, size=n_reservoir)
    return input_weights, reservoir_weights, bias


def esn_state(
    sequence: np.ndarray,
    input_weights: np.ndarray,
    reservoir_weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Return the locked final leaky ESN state for one input sequence."""

    state = np.zeros(reservoir_weights.shape[0])
    for observation in sequence:
        state = LEAK_RETAIN * state + LEAK_UPDATE * np.tanh(
            input_weights @ observation + reservoir_weights @ state + bias
        )
    return state


def fit_esn_predict(
    train_sequences: np.ndarray,
    y: np.ndarray,
    test_sequence: np.ndarray,
    input_weights: np.ndarray,
    reservoir_weights: np.ndarray,
    bias: np.ndarray,
    C: float = 0.1,
) -> float:
    """Scale sequence inputs, compute fixed ESN states, and predict one row."""

    scaler = StandardScaler()
    scaler.fit(train_sequences.reshape(-1, train_sequences.shape[-1]))
    train_states = np.vstack([
        esn_state(scaler.transform(sequence), input_weights, reservoir_weights, bias)
        for sequence in train_sequences
    ])
    test_state = np.vstack([
        esn_state(scaler.transform(test_sequence), input_weights, reservoir_weights, bias)
    ])
    model = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    model.fit(train_states, y)
    return float(model.predict_proba(test_state)[0, 1])
