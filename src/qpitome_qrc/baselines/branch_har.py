"""Causal HAR-derived features for branch-resolution baselines.

This preserves the historical canonical HAR formulation exactly:

    [rv_5d, rv_10d, rv_20d, rv_60d, vix_close]
    -> StandardScaler
    -> Ridge(alpha=1.0)
    -> future_rv_20d

For each branch episode, the HAR model is fit only on daily rows whose 20-day
future target is fully observable before that branch point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HAR_FEATURES = ("rv_5d", "rv_10d", "rv_20d", "rv_60d", "vix_close")
HAR_TARGET = "future_rv_20d"
HAR_TARGET_HORIZON_ROWS = 20


def add_causal_har_episode_features(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    alpha: float = 1.0,
    target_horizon_rows: int = HAR_TARGET_HORIZON_ROWS,
    prediction_floor: float = 1e-8,
    min_daily_train_rows: int = 504,
) -> pd.DataFrame:
    """Attach one causal HAR forecast and derived diagnostics per episode."""
    required_daily = {HAR_TARGET, *HAR_FEATURES}
    missing = required_daily - set(daily.columns)
    if missing:
        raise KeyError(f"Missing HAR daily columns: {sorted(missing)}")
    if "branch_idx" not in episodes.columns:
        raise KeyError("branch_idx")

    out = episodes.copy()
    pred_values: list[float] = []
    n_train_values: list[int] = []
    latest_train_idx_values: list[int] = []

    x_all = daily.loc[:, HAR_FEATURES].to_numpy(dtype=float)
    y_all = daily[HAR_TARGET].to_numpy(dtype=float)

    for row in out.itertuples(index=False):
        branch_idx = int(row.branch_idx)
        latest_train_idx = branch_idx - target_horizon_rows - 1
        if latest_train_idx < 0:
            raise ValueError(f"No leakage-safe HAR training rows for branch_idx={branch_idx}")

        train_slice = slice(0, latest_train_idx + 1)
        finite_train = np.isfinite(y_all[train_slice]) & np.isfinite(x_all[train_slice]).all(axis=1)
        x_train = x_all[train_slice][finite_train]
        y_train = y_all[train_slice][finite_train]

        if len(y_train) < min_daily_train_rows:
            raise ValueError(
                f"Episode {row.episode_id} has only {len(y_train)} HAR training rows; "
                f"minimum is {min_daily_train_rows}"
            )
        x_test = x_all[[branch_idx]]
        if not np.isfinite(x_test).all():
            raise ValueError(f"Non-finite HAR inputs at branch_idx={branch_idx}")

        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_train, y_train)
        pred = max(float(model.predict(x_test)[0]), prediction_floor)

        pred_values.append(pred)
        n_train_values.append(len(y_train))
        latest_train_idx_values.append(latest_train_idx)

    out["har_future_rv20_pred"] = pred_values
    out["har_daily_train_rows"] = n_train_values
    out["har_latest_target_row"] = latest_train_idx_values
    out["har_log_future_to_current"] = np.log(
        out["har_future_rv20_pred"].to_numpy(dtype=float)
        / out["rv_20d"].to_numpy(dtype=float)
    )
    out["har_future_minus_current"] = (
        out["har_future_rv20_pred"].to_numpy(dtype=float)
        - out["rv_20d"].to_numpy(dtype=float)
    )
    out["har_future_over_vix"] = (
        out["har_future_rv20_pred"].to_numpy(dtype=float)
        / out["vix_close"].to_numpy(dtype=float)
    )
    return out
