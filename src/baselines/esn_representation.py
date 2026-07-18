"""Input transforms and trajectory pooling for ESN representation studies."""

from __future__ import annotations

import numpy as np


INPUT_REPRESENTATIONS = ("level", "level_diff", "level_diff_time")
POOLINGS = ("final", "mean", "std", "final_mean_std")


def transform_input(X: np.ndarray, representation: str) -> np.ndarray:
    """Build compact channels from a univariate sequence tensor."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 1:
        raise ValueError(f"expected X shaped (samples, steps, 1), got {values.shape}")
    level = values[..., 0]
    diff = np.diff(level, axis=1, prepend=level[:, :1])
    time = np.linspace(0.0, 1.0, level.shape[1], dtype=float)[None, :]
    time = np.repeat(time, level.shape[0], axis=0)
    if representation == "level":
        return level[..., None]
    if representation == "level_diff":
        return np.stack([level, diff], axis=-1)
    if representation == "level_diff_time":
        return np.stack([level, diff, time], axis=-1)
    raise ValueError(f"unknown input representation: {representation}")


def reservoir_trajectory(
    X: np.ndarray,
    W_in: np.ndarray,
    W: np.ndarray,
    leak: float,
) -> np.ndarray:
    """Return every reservoir state for each independently reset input window."""
    values = np.asarray(X, dtype=float)
    states = np.empty((values.shape[0], values.shape[1], W.shape[0]), dtype=float)
    for sample, window in enumerate(values):
        h = np.zeros(W.shape[0], dtype=float)
        for step, u_t in enumerate(window):
            h_new = np.tanh(W_in @ u_t + W @ h)
            h = (1.0 - leak) * h + leak * h_new
            states[sample, step] = h
    return states


def pool_trajectory(states: np.ndarray, pooling: str, washout: int = 0) -> np.ndarray:
    """Pool post-washout reservoir trajectories into fixed-width features."""
    values = np.asarray(states, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"expected states shaped (samples, steps, units), got {values.shape}")
    if washout < 0 or washout >= values.shape[1]:
        raise ValueError("washout must satisfy 0 <= washout < sequence length")
    kept = values[:, washout:, :]
    final = kept[:, -1, :]
    mean = kept.mean(axis=1)
    std = kept.std(axis=1)
    if pooling == "final":
        return final
    if pooling == "mean":
        return mean
    if pooling == "std":
        return std
    if pooling == "final_mean_std":
        return np.concatenate([final, mean, std], axis=1)
    raise ValueError(f"unknown pooling: {pooling}")
