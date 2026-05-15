from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_tabular_split(df: pd.DataFrame, date_col: str = "date", train_end: str = "2016-01-01", val_end: str = "2020-01-01") -> dict[str, pd.DataFrame]:
    """Return chronological train/validation/test DataFrame splits."""
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    return {
        "train": out[out[date_col] < train_end_dt].copy(),
        "val": out[(out[date_col] >= train_end_dt) & (out[date_col] < val_end_dt)].copy(),
        "test": out[out[date_col] >= val_end_dt].copy(),
    }


def split_arrays(splits: dict[str, pd.DataFrame], features: list[str], target: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert split DataFrames into X/y arrays."""
    required = set(features + [target])
    arrays = {}

    for name, split in splits.items():
        missing = required - set(split.columns)
        if missing:
            raise ValueError(f"{name} split missing columns: {sorted(missing)}")
        arrays[name] = (split[features].to_numpy(dtype=float), split[target].to_numpy(dtype=int))

    return arrays


def describe_splits(splits: dict[str, pd.DataFrame], target: str, date_col: str = "date") -> pd.DataFrame:
    """Return a compact split summary table."""
    rows = []
    for name, split in splits.items():
        y = split[target].astype(int)
        rows.append({
            "split": name,
            "n_rows": len(split),
            "date_min": split[date_col].min(),
            "date_max": split[date_col].max(),
            "n_class_0": int((y == 0).sum()),
            "n_class_1": int((y == 1).sum()),
            "class_1_rate": float((y == 1).mean()),
        })
    return pd.DataFrame(rows)
