"""Shared feature transforms and diagnostics for day-5 assays."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PATH_SHAPE_EPS = 1e-9
PATH_SHAPE_BLOCKS: dict[str, list[str]] = {
    "trajectory": [f"trajectory_r_d{day}" for day in range(1, 5)],
    "shape": [
        "path_total_variation",
        "path_efficiency",
        "path_curvature_l1",
        "path_reversal_count",
        "path_argmin_day",
        "path_argmax_day",
        "path_early_late_imbalance",
    ],
    "barrier_path": [
        "path_min_relapse_distance_scaled",
        "path_min_recovery_distance_scaled",
        "path_relapse_dwell25",
        "path_recovery_dwell25",
        "path_relapse_first_day",
        "path_recovery_first_day",
    ],
}
PATH_SHAPE_BLOCKS["all_path_shape"] = (
    PATH_SHAPE_BLOCKS["trajectory"]
    + PATH_SHAPE_BLOCKS["shape"]
    + PATH_SHAPE_BLOCKS["barrier_path"]
)


def add_path_shape_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the locked trajectory, shape, and barrier-path audit features."""

    out = frame.copy()
    path = np.column_stack([
        np.zeros(len(out), dtype=float),
        *[out[f"r_d{day}"].to_numpy(float) for day in range(1, 6)],
    ])
    increments = np.diff(path, axis=1)
    second = np.diff(path, n=2, axis=1)

    endpoint = out["current_return_5d_from_branch"].to_numpy(float)
    lower = endpoint - out["distance_to_relapse_barrier"].to_numpy(float)
    upper = endpoint + out["distance_to_recovery_barrier"].to_numpy(float)
    width = np.maximum(out["barrier_width"].to_numpy(float), PATH_SHAPE_EPS)
    running = path[:, 1:]

    out["path_total_variation"] = np.sum(np.abs(increments), axis=1)
    out["path_efficiency"] = np.abs(endpoint) / np.maximum(
        out["path_total_variation"].to_numpy(float), PATH_SHAPE_EPS
    )
    out["path_curvature_l1"] = np.sum(np.abs(second), axis=1)
    out["path_reversal_count"] = np.sum(
        increments[:, 1:] * increments[:, :-1] < 0, axis=1
    )
    out["path_argmin_day"] = np.argmin(running, axis=1) + 1
    out["path_argmax_day"] = np.argmax(running, axis=1) + 1
    out["path_early_late_imbalance"] = (
        np.sum(np.abs(increments[:, :2]), axis=1)
        - np.sum(np.abs(increments[:, 3:]), axis=1)
    )

    relapse_dist = (running - lower[:, None]) / width[:, None]
    recovery_dist = (upper[:, None] - running) / width[:, None]
    out["path_min_relapse_distance_scaled"] = relapse_dist.min(axis=1)
    out["path_min_recovery_distance_scaled"] = recovery_dist.min(axis=1)
    out["path_relapse_dwell25"] = np.sum(relapse_dist <= 0.25, axis=1)
    out["path_recovery_dwell25"] = np.sum(recovery_dist <= 0.25, axis=1)
    out["path_relapse_first_day"] = np.argmin(relapse_dist, axis=1) + 1
    out["path_recovery_first_day"] = np.argmin(recovery_dist, axis=1) + 1

    for day in range(1, 5):
        out[f"trajectory_r_d{day}"] = out[f"r_d{day}"].to_numpy(float)
    return out


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
