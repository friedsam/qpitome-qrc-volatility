"""Shared locked protocol for the day-5 branching experiments.

This module contains data preparation and eligibility rules that were originally
embedded in a shard runner. Keeping them in package code prevents analysis
scripts from depending on one another by file path while preserving the exact
historical experiment definitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.qrc.local_detuning_reservoir import LocalDetuningConfig
from qpitome_qrc.qrc.rydberg_reservoir import RydbergQRCConfig

MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")

D1 = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "barrier_width",
]

STATIC = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "closest_to_relapse",
    "closest_to_recovery",
]


def add_extrema(frame: pd.DataFrame) -> pd.DataFrame:
    """Add locked closest-barrier path features without mutating ``frame``."""

    out = frame.copy()
    running = np.column_stack([out[f"r_d{day}"].to_numpy(float) for day in range(1, 6)])
    endpoint = out["current_return_5d_from_branch"].to_numpy(float)
    lower = endpoint - out["distance_to_relapse_barrier"].to_numpy(float)
    upper = endpoint + out["distance_to_recovery_barrier"].to_numpy(float)
    out["closest_to_relapse"] = running.min(axis=1) - lower
    out["closest_to_recovery"] = upper - running.max(axis=1)
    return out


def load_frame(path_panel: Path, clusters_path: Path) -> pd.DataFrame:
    """Load the locked day-5 panel and attach synchronized crisis clusters."""

    frame = pd.read_csv(path_panel, parse_dates=["branch_date", "landmark_date"])
    clusters = pd.read_csv(clusters_path)[["market_key", "episode_id", "cluster_id"]].drop_duplicates()
    frame = frame.merge(clusters, on=["market_key", "episode_id"], how="left", validate="many_to_one")
    if frame["cluster_id"].isna().any():
        raise ValueError("path-panel rows are missing cluster ids")

    starts = (
        frame.groupby("cluster_id", as_index=False)["branch_date"]
        .min()
        .rename(columns={"branch_date": "cluster_start"})
    )
    frame = frame.merge(starts, on="cluster_id", how="left", validate="many_to_one")
    frame = add_extrema(frame)

    needed = D1 + STATIC + [f"r_d{day}" for day in range(1, 6)] + [
        "y_recovery",
        "landmark_date",
        "cluster_start",
    ]
    frame = (
        frame.dropna(subset=needed)
        .sort_values(["landmark_date", "market_key", "episode_id"])
        .reset_index(drop=True)
    )
    if not np.allclose(
        frame["r_d5"].to_numpy(float),
        frame["current_return_5d_from_branch"].to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("r_d5 does not match locked day-5 endpoint return")
    return frame


def differential_patterns(train_X: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Map five standardized static inputs to paired positive/negative channels."""

    scaler = StandardScaler().fit(train_X)
    z = scaler.transform(X)
    positive = 0.5 * (1.0 + np.tanh(z))
    negative = 1.0 - positive
    patterns = np.empty((len(z), 10), dtype=float)
    patterns[:, 0::2] = positive
    patterns[:, 1::2] = negative
    return patterns


def rydberg_config() -> LocalDetuningConfig:
    """Return the locked local-detuning Rydberg configuration for this assay."""

    reservoir = RydbergQRCConfig(
        geometry="chain",
        chain_atoms=10,
        chain_spacing_um=7.5,
        observable_mode="n_nn",
        collect_anchor_features=False,
        memory_mode="memoryless",
        shots=None,
    )
    return LocalDetuningConfig(
        reservoir=reservoir,
        evolution_time_us=0.55,
        global_omega_rad_us=6.0,
        global_delta_rad_us=6.0,
        local_delta_rad_us=4.0,
    )


def eligible_rows(frame: pd.DataFrame) -> list[int]:
    """Return rows satisfying the locked prequential historical-training rule."""

    result: list[int] = []
    for i, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train = frame[frame["landmark_date"] < row["cluster_start"]]
        if len(train) >= MIN_TRAIN and train["y_recovery"].nunique() >= 2:
            result.append(i)
    return result
