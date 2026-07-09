"""Matched causal path representations for branch-resolution diagnostics.

The historical ESN module remains untouched. This module reuses its exact
reservoir update mechanics while adding a continuous-state representation and a
matched flattened-path control for the branch problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.numpy_esn import esn_states, make_esn_weights


PATH_COLUMNS = (
    "spy_log_return",
    "spy_abs_log_return",
    "log_rv5_over_rv20",
    "log_rv20_over_rv60",
    "drawdown_120d_path",
    "rv5_change_scaled",
)


def add_causal_path_channels(daily: pd.DataFrame) -> pd.DataFrame:
    """Construct dimensionless causal daily channels without fitted preprocessing."""
    required = {"spy_log_return", "rv_5d", "rv_20d", "rv_60d", "spy_adj_close"}
    missing = required - set(daily.columns)
    if missing:
        raise KeyError(f"Missing path inputs: {sorted(missing)}")

    out = daily.copy().sort_values("date").reset_index(drop=True)
    out["spy_abs_log_return"] = out["spy_log_return"].abs()
    out["log_rv5_over_rv20"] = np.log(out["rv_5d"] / out["rv_20d"])
    out["log_rv20_over_rv60"] = np.log(out["rv_20d"] / out["rv_60d"])
    peak_120 = out["spy_adj_close"].rolling(120).max()
    out["drawdown_120d_path"] = out["spy_adj_close"] / peak_120 - 1.0
    out["rv5_change_scaled"] = (out["rv_5d"] - out["rv_5d"].shift(5)) / out["rv_20d"]

    # Fixed clipping prevents isolated historical extremes from saturating the
    # reservoir while avoiding train/test-fitted preprocessing.
    clip_bounds = {
        "spy_log_return": (-0.15, 0.15),
        "spy_abs_log_return": (0.0, 0.15),
        "log_rv5_over_rv20": (-2.0, 2.0),
        "log_rv20_over_rv60": (-1.5, 1.5),
        "drawdown_120d_path": (-0.60, 0.05),
        "rv5_change_scaled": (-3.0, 3.0),
    }
    for col, (lo, hi) in clip_bounds.items():
        out[col] = out[col].clip(lo, hi)
    return out


def extract_episode_windows(
    daily_channels: pd.DataFrame,
    episodes: pd.DataFrame,
    lookback: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Return episode ids and inclusive 40-day paths ending at each branch point."""
    values = daily_channels.loc[:, PATH_COLUMNS].to_numpy(dtype=float)
    ids: list[int] = []
    windows: list[np.ndarray] = []
    for row in episodes.itertuples(index=False):
        branch_idx = int(row.branch_idx)
        start = branch_idx - lookback + 1
        if start < 0:
            continue
        window = values[start : branch_idx + 1]
        if len(window) != lookback or not np.isfinite(window).all():
            continue
        ids.append(int(row.episode_id))
        windows.append(window)
    return np.asarray(ids, dtype=int), np.asarray(windows, dtype=float)


def flattened_path_features(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(len(windows), -1)


def reset_esn_features(
    windows: np.ndarray,
    *,
    n_reservoir: int,
    spectral_radius: float,
    input_scale: float,
    leak: float,
    seed: int,
) -> np.ndarray:
    """Historical zero-reset local-window ESN representation."""
    w_in, w = make_esn_weights(
        n_inputs=windows.shape[2],
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        input_scale=input_scale,
        seed=seed,
    )
    return esn_states(windows, w_in, w, leak)


def continuous_esn_daily_states(
    daily_channels: pd.DataFrame,
    *,
    n_reservoir: int,
    spectral_radius: float,
    input_scale: float,
    leak: float,
    seed: int,
) -> np.ndarray:
    """Run one reservoir continuously after the causal channel warm-up period."""
    x = daily_channels.loc[:, PATH_COLUMNS].to_numpy(dtype=float)
    finite_rows = np.isfinite(x).all(axis=1)
    if not finite_rows.any():
        raise ValueError("Continuous ESN channels contain no fully finite rows")

    first_finite = int(np.flatnonzero(finite_rows)[0])
    if not finite_rows[first_finite:].all():
        bad = np.flatnonzero(~finite_rows[first_finite:]) + first_finite
        raise ValueError(
            "Continuous ESN channels contain non-finite values after warm-up: "
            f"first bad row {int(bad[0])}"
        )

    w_in, w = make_esn_weights(
        n_inputs=x.shape[1],
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        input_scale=input_scale,
        seed=seed,
    )
    h = np.zeros(n_reservoir, dtype=float)
    rows = np.full((len(x), n_reservoir + x.shape[1]), np.nan, dtype=float)
    for row_idx in range(first_finite, len(x)):
        u_t = x[row_idx]
        h_new = np.tanh(w_in @ u_t + w @ h)
        h = (1.0 - leak) * h + leak * h_new
        rows[row_idx] = np.concatenate([h.copy(), u_t])
    return rows


def episode_features_from_daily_states(
    daily_states: np.ndarray,
    episodes: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    ids = episodes["episode_id"].astype(int).to_numpy()
    idx = episodes["branch_idx"].astype(int).to_numpy()
    features = daily_states[idx]
    if not np.isfinite(features).all():
        raise ValueError("One or more episode branch points occur before ESN warm-up completes")
    return ids, features
