from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TRAIN_END = "2014-12-31"
DEFAULT_VAL_END = "2019-12-31"


def chronological_tabular_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = DEFAULT_TRAIN_END,
    val_end: str = DEFAULT_VAL_END,
) -> dict[str, pd.DataFrame]:
    """Return chronological train/validation/test DataFrame splits.

    Boundaries are inclusive for train and validation end dates:
      train: date <= train_end
      val:   train_end < date <= val_end
      test:  date > val_end
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    return {
        "train": out[out[date_col] <= train_end_dt].copy(),
        "val": out[(out[date_col] > train_end_dt) & (out[date_col] <= val_end_dt)].copy(),
        "test": out[out[date_col] > val_end_dt].copy(),
    }


def split_arrays(
    splits: dict[str, pd.DataFrame],
    features: list[str],
    target: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert split DataFrames into X/y arrays for regression or classification."""
    required = set(features + [target])
    arrays = {}

    for name, split in splits.items():
        missing = required - set(split.columns)
        if missing:
            raise ValueError(f"{name} split missing columns: {sorted(missing)}")
        arrays[name] = (
            split[features].to_numpy(dtype=float),
            split[target].to_numpy(dtype=float),
        )

    return arrays


def describe_regression_splits(
    splits: dict[str, pd.DataFrame],
    targets: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Return split summary for continuous volatility targets."""
    rows = []
    for name, split in splits.items():
        row = {
            "split": name,
            "n_rows": len(split),
            "date_min": split[date_col].min(),
            "date_max": split[date_col].max(),
        }
        for target in targets:
            y = split[target].astype(float)
            row[f"{target}_mean"] = float(y.mean())
            row[f"{target}_std"] = float(y.std())
            row[f"{target}_q50"] = float(y.quantile(0.50))
            row[f"{target}_q80"] = float(y.quantile(0.80))
        rows.append(row)
    return pd.DataFrame(rows)


def add_train_only_transition_flags(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    train_end: str = DEFAULT_TRAIN_END,
    current_vol_col: str = "rv_20d",
    future_vol_col: str = "future_rv_20d",
    calm_quantile: float = 0.50,
    turbulent_quantile: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add transparent regime-transition flags using train-only thresholds."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    train = out[out[date_col] <= pd.Timestamp(train_end)]
    if train.empty:
        raise ValueError("Training split is empty; cannot compute train-only thresholds.")

    calm_threshold = float(train[current_vol_col].quantile(calm_quantile))
    turbulent_threshold = float(train[future_vol_col].quantile(turbulent_quantile))

    out["calm_now"] = out[current_vol_col] <= calm_threshold
    out["future_turbulent"] = out[future_vol_col] >= turbulent_threshold
    out["transition_event"] = out["calm_now"] & out["future_turbulent"]

    thresholds = {
        "calm_threshold": calm_threshold,
        "turbulent_threshold": turbulent_threshold,
        "calm_quantile": calm_quantile,
        "turbulent_quantile": turbulent_quantile,
    }
    return out, thresholds


def describe_transition_flags(
    splits: dict[str, pd.DataFrame],
    event_col: str = "transition_event",
    date_col: str = "date",
) -> pd.DataFrame:
    """Summarize sparse transition-event flags by split."""
    rows = []
    for name, split in splits.items():
        event = split[event_col].astype(bool)
        rows.append(
            {
                "split": name,
                "n_rows": len(split),
                "date_min": split[date_col].min(),
                "date_max": split[date_col].max(),
                "n_events": int(event.sum()),
                "event_rate": float(event.mean()),
            }
        )
    return pd.DataFrame(rows)


# Backward-compatible alias for older classification utilities.
def describe_splits(
    splits: dict[str, pd.DataFrame],
    target: str,
    date_col: str = "date",
) -> pd.DataFrame:
    if target in {"future_rv_5d", "future_rv_20d"}:
        return describe_regression_splits(splits, [target], date_col=date_col)

    rows = []
    for name, split in splits.items():
        y = split[target].astype(int)
        rows.append(
            {
                "split": name,
                "n_rows": len(split),
                "date_min": split[date_col].min(),
                "date_max": split[date_col].max(),
                "n_class_0": int((y == 0).sum()),
                "n_class_1": int((y == 1).sum()),
                "class_1_rate": float((y == 1).mean()),
            }
        )
    return pd.DataFrame(rows)
