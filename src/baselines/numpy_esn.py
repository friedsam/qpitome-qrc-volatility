"""Deterministic NumPy echo-state reservoir used by the Phase 2/3 benchmarks.

This module owns only the reservoir mechanics and ridge readouts. Evaluation
geometry, preprocessing, model selection, and reporting belong to experiment
runners.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SPLIT_NAMES = ("train", "val", "test")


def spectral_scale(W: np.ndarray, radius: float) -> np.ndarray:
    """Rescale a recurrent matrix to the requested spectral radius."""
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W * (radius / max(rho, 1e-12))


def make_esn_weights(
    n_inputs: int,
    n_reservoir: int,
    spectral_radius: float,
    input_scale: float,
    seed: int,
    connectivity: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic ESN input/recurrent weights.

    ``connectivity`` is the Bernoulli probability that a recurrent edge is
    retained. The historical default remains 0.10.
    """
    if not 0.0 < connectivity <= 1.0:
        raise ValueError("connectivity must lie in (0, 1]")

    rng = np.random.default_rng(seed)
    W_in = rng.normal(
        0.0,
        input_scale,
        size=(n_reservoir, n_inputs),
    )
    W = rng.normal(
        0.0,
        1.0,
        size=(n_reservoir, n_reservoir),
    )
    W *= rng.random(W.shape) < connectivity

    # Extremely sparse draws can contain no recurrent edge. Preserve a valid
    # deterministic reservoir rather than allowing spectral scaling to collapse.
    if not np.any(W):
        W[rng.integers(0, n_reservoir), rng.integers(0, n_reservoir)] = 1.0

    return W_in, spectral_scale(W, spectral_radius)


def esn_states(
    X: np.ndarray,
    W_in: np.ndarray,
    W: np.ndarray,
    leak: float,
) -> np.ndarray:
    """Convert local input windows into final-state-plus-final-input features.

    The reservoir state is reset to zero for every input window. This preserves
    the exact historical NumPy ESN behavior.
    """
    rows = []

    for window in X:
        h = np.zeros(W.shape[0])

        for u_t in window:
            h_new = np.tanh(W_in @ u_t + W @ h)
            h = (1.0 - leak) * h + leak * h_new

        rows.append(np.concatenate([h, window[-1]]))

    return np.asarray(rows)


def fit_continuous_ridge_scores(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    score_features: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    """Fit a train-only standardized multi-output ridge readout.

    Targets are used exactly as supplied. In particular, callers with targets
    that are already log volatility must not apply another logarithm.
    """
    scaler = StandardScaler()
    model = Ridge(alpha=alpha)
    model.fit(scaler.fit_transform(train_features), np.asarray(train_targets, dtype=float))
    return {
        name: model.predict(scaler.transform(values))
        for name, values in score_features.items()
    }


def fit_log_ridge_scores(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    """Fit the historical standardized ridge readout on positive volatility.

    This compatibility function preserves the original Phase 2/3 behavior.
    New already-log targets should use ``fit_continuous_ridge_scores``.
    """
    return fit_continuous_ridge_scores(
        features["train"],
        np.log(np.maximum(targets["train"], 1e-8)),
        {split: features[split] for split in SPLIT_NAMES},
        alpha,
    )


def historical_numpy_esn_grid(seeds: list[int]) -> list[dict]:
    """Return the four frozen NumPy ESN configurations used in Phase 2/3.

    This is a historical reference grid, not the search space for future
    Phase 3 tuning.
    """
    base = [
        {"n": 300, "sr": 0.70, "inp": 0.30, "leak": 0.30, "alpha": 300.0},
        {"n": 300, "sr": 0.90, "inp": 0.30, "leak": 0.30, "alpha": 1000.0},
        {"n": 500, "sr": 0.70, "inp": 0.20, "leak": 0.50, "alpha": 1000.0},
        {"n": 500, "sr": 0.90, "inp": 0.20, "leak": 0.50, "alpha": 3000.0},
    ]

    grid = []
    for config in base:
        for seed in seeds:
            row = dict(config, seed=seed)
            row["config_id"] = (
                f"esn_n{row['n']}_sr{row['sr']}_inp{row['inp']}_"
                f"leak{row['leak']}_alpha{row['alpha']}_seed{seed}"
            )
            grid.append(row)

    return grid
