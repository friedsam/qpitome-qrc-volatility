"""Shared feature transforms and diagnostics for day-5 Rydberg assays."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def split_blocks(features: np.ndarray) -> dict[str, np.ndarray]:
    """Split occupation/pair observables and derive connected pair features."""

    occ = features[:, :10]
    raw_pairs = features[:, 10:]
    connected = np.empty_like(raw_pairs)
    k = 0
    for i in range(9):
        for j in range(i + 1, 10):
            connected[:, k] = raw_pairs[:, k] - occ[:, i] * occ[:, j]
            k += 1
    return {
        "occupations": occ,
        "raw_pairs": raw_pairs,
        "connected_pairs": connected,
        "all_raw": features,
        "occ_plus_connected": np.column_stack([occ, connected]),
    }


def feature_diagnostics(X: np.ndarray) -> dict[str, float]:
    """Return the locked rank, conditioning, and near-constant diagnostics."""

    X = np.asarray(X, dtype=float)
    centered = X - X.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    total = float(variance.sum())
    if total <= 1e-15:
        return {
            "n_features": int(X.shape[1]),
            "participation_ratio": 0.0,
            "n95": 0,
            "condition_number": np.inf,
            "near_constant": int(X.shape[1]),
        }
    participation = total**2 / float(np.sum(variance**2))
    n95 = int(np.searchsorted(np.cumsum(variance) / total, 0.95) + 1)
    nonzero = singular[singular > 1e-12]
    condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else np.inf
    near_constant = int(np.sum(np.std(X, axis=0) < 1e-8))
    return {
        "n_features": int(X.shape[1]),
        "participation_ratio": float(participation),
        "n95": n95,
        "condition_number": condition,
        "near_constant": near_constant,
    }


def pca_block(
    H_train: np.ndarray,
    H_test: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize a feature block and apply the locked full-SVD PCA transform."""

    scaler = StandardScaler().fit(H_train)
    train_scaled = scaler.transform(H_train)
    test_scaled = scaler.transform(H_test)
    actual_rank = min(rank, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=actual_rank, svd_solver="full").fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(test_scaled)
