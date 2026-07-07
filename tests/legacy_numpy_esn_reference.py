"""Frozen pre-refactor NumPy ESN implementation.

This file intentionally duplicates the historical implementation from
run_phase3_esn_ridge_walkforward.py. It must not import production ESN code.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SPLIT_NAMES = ("train", "val", "test")


def legacy_spectral_scale(W, radius):
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W * (radius / max(rho, 1e-12))


def legacy_make_esn_weights(
    n_inputs,
    n_reservoir,
    spectral_radius,
    input_scale,
    seed,
):
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
    W *= rng.random(W.shape) < 0.10

    return W_in, legacy_spectral_scale(W, spectral_radius)


def legacy_esn_states(X, W_in, W, leak):
    rows = []

    for window in X:
        h = np.zeros(W.shape[0])

        for u_t in window:
            h_new = np.tanh(W_in @ u_t + W @ h)
            h = (1.0 - leak) * h + leak * h_new

        rows.append(np.concatenate([h, window[-1]]))

    return np.asarray(rows)


def legacy_fit_ridge_scores(H, y, alpha):
    scaler = StandardScaler()
    model = Ridge(alpha=alpha)

    model.fit(
        scaler.fit_transform(H["train"]),
        np.log(np.maximum(y["train"], 1e-8)),
    )

    return {
        split: model.predict(scaler.transform(H[split]))
        for split in SPLIT_NAMES
    }
