"""Task-aligned causal path channels for branch-transition forecasting.

These channels are deliberately different from the generic six-channel ESN path.
They focus on temporal ordering that is not explicitly preserved by the
branch-point ``state_plus_motion`` baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRANSITION_PATH_COLUMNS = (
    "signed_return_over_local_vol",
    "downside_shock_pressure",
    "d_log_rv5_over_rv20",
    "drawdown_repair_over_local_vol",
)

TRANSITION_PATH_CLIP_BOUNDS = {
    "signed_return_over_local_vol": (-4.0, 4.0),
    "downside_shock_pressure": (0.0, 1.0),
    "d_log_rv5_over_rv20": (-1.0, 1.0),
    "drawdown_repair_over_local_vol": (-4.0, 4.0),
}


def add_transition_path_channels(daily: pd.DataFrame) -> pd.DataFrame:
    """Construct four fixed causal transition-specific daily channels.

    The channels encode:

    1. signed return normalized by contemporaneous local volatility;
    2. recent downside shock energy relative to total 20-day return energy;
    3. one-day change in the short/medium realized-volatility ratio;
    4. drawdown repair increment normalized by local volatility.

    No fitted preprocessing and no future information are used.
    """
    required = {"date", "spy_log_return", "rv_5d", "rv_20d", "spy_adj_close"}
    missing = required - set(daily.columns)
    if missing:
        raise KeyError(f"Missing transition-path inputs: {sorted(missing)}")

    out = daily.copy().sort_values("date").reset_index(drop=True)

    # Project RV measures are annualized. Convert the 20-day level to an
    # approximate one-day scale before normalizing daily returns.
    local_daily_vol = out["rv_20d"] / np.sqrt(252.0)
    safe_daily_vol = local_daily_vol.where(local_daily_vol > 0)

    out["signed_return_over_local_vol"] = (
        out["spy_log_return"] / safe_daily_vol
    )

    negative_energy = out["spy_log_return"].clip(upper=0.0).pow(2)
    total_energy = out["spy_log_return"].pow(2)
    downside_5 = negative_energy.rolling(5, min_periods=5).sum()
    total_20 = total_energy.rolling(20, min_periods=20).sum()
    out["downside_shock_pressure"] = downside_5 / total_20.where(total_20 > 0)

    log_ratio = np.log(out["rv_5d"] / out["rv_20d"])
    out["d_log_rv5_over_rv20"] = log_ratio.diff()

    peak_120 = out["spy_adj_close"].rolling(120, min_periods=120).max()
    drawdown = out["spy_adj_close"] / peak_120 - 1.0
    out["drawdown_repair_over_local_vol"] = drawdown.diff() / safe_daily_vol

    for column, (lower, upper) in TRANSITION_PATH_CLIP_BOUNDS.items():
        out[column] = out[column].clip(lower, upper)

    return out


def extract_transition_windows(
    daily_channels: pd.DataFrame,
    episodes: pd.DataFrame,
    lookback: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete causal windows ending at each episode branch point."""
    missing = set(TRANSITION_PATH_COLUMNS) - set(daily_channels.columns)
    if missing:
        raise KeyError(f"Missing transition path channels: {sorted(missing)}")
    if lookback < 1:
        raise ValueError("lookback must be positive")

    values = daily_channels.loc[:, TRANSITION_PATH_COLUMNS].to_numpy(dtype=float)
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
