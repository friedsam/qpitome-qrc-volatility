import numpy as np
import pandas as pd


def make_rolling_sequences(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    seq_len: int,
    date_col: str = "date",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert tabular time series into rolling sequence samples."""
    missing = set(features + [target, date_col]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    X_raw = df[features].to_numpy(dtype=float)
    y_raw = df[target].to_numpy(dtype=int)
    dates = df[date_col].to_numpy()

    X, y, out_dates = [], [], []

    for i in range(seq_len - 1, len(df)):
        X.append(X_raw[i - seq_len + 1 : i + 1])
        y.append(y_raw[i])
        out_dates.append(dates[i])

    return np.asarray(X), np.asarray(y), np.asarray(out_dates)


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_end: str = "2016-01-01",
    val_end: str = "2020-01-01",
):
    """Chronological train/validation/test split."""
    train_end_dt = np.datetime64(train_end)
    val_end_dt = np.datetime64(val_end)

    train_mask = dates < train_end_dt
    val_mask = (dates >= train_end_dt) & (dates < val_end_dt)
    test_mask = dates >= val_end_dt

    return {
        "train": (X[train_mask], y[train_mask], dates[train_mask]),
        "val": (X[val_mask], y[val_mask], dates[val_mask]),
        "test": (X[test_mask], y[test_mask], dates[test_mask]),
    }