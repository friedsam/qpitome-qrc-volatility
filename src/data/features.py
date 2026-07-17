from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


FEATURE_COLUMNS = [
    # SPY return / range state
    "spy_log_return",
    "spy_abs_log_return",
    "spy_squared_log_return",
    "spy_log_hl_range",
    "spy_log_oc_return",
    # SPY volume / liquidity state
    "spy_log_volume",
    "spy_log_volume_change",
    "spy_log_dollar_volume",
    # trailing volatility state
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "rv_60d",
    "rv_ratio_5_20",
    "rv_ratio_20_60",
    "rv_slope_5_20",
    "rv_slope_20_60",
    # daily OHLC variance proxies
    "parkinson_var",
    "garman_klass_var",
    # drawdown / market stress state
    "spy_drawdown_20d",
    # VIX state
    "vix_close",
    "vix_log_change",
    "vix_abs_log_change",
    "vix_log_hl_range",
    "vix_ma_5d",
    "vix_std_5d",
    "vix_ma_20d",
    "vix_std_20d",
]

TARGET_COLUMNS = [
    "future_rv_5d",
    "future_rv_20d",
]

DEFAULT_FEATURE_COLUMNS = FEATURE_COLUMNS
DEFAULT_TARGET_COLUMNS = TARGET_COLUMNS


@dataclass
class RegressionArrays:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str]
    target_column: str
    scaler_name: str


def validate_columns(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
) -> None:
    """Validate that a Phase 2 modeling frame contains required columns."""
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    target_columns = target_columns or DEFAULT_TARGET_COLUMNS

    required = set(feature_columns + target_columns + ["date"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required Phase 2 columns: {sorted(missing)}")


def drop_nonfinite_model_rows(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop rows with NaN/inf in selected features or targets."""
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    target_columns = target_columns or DEFAULT_TARGET_COLUMNS
    validate_columns(df, feature_columns=feature_columns, target_columns=target_columns)

    out = df.copy()
    cols = list(dict.fromkeys(feature_columns + target_columns))
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=cols).reset_index(drop=True)


def make_scaler(name: str = "standard"):
    """Construct a scaler for feature normalization.

    standard: zero mean, unit variance; default for linear/ESN/QRC prototypes.
    robust: median/IQR scaling; useful if volatility spikes dominate too strongly.
    none: no scaling.
    """
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name in {"none", None}:
        return None
    raise ValueError(f"Unknown scaler: {name}")


def scale_splits_train_only(
    splits: dict[str, pd.DataFrame],
    *,
    feature_columns: list[str] | None = None,
    scaler_name: str = "standard",
) -> tuple[dict[str, pd.DataFrame], object | None]:
    """Scale feature columns using train-fit statistics only.

    This enforces the Phase 2 leakage rule: validation/test statistics must not
    influence normalization.
    """
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS

    if "train" not in splits:
        raise ValueError("splits must contain a 'train' split")

    scaler = make_scaler(scaler_name)
    scaled = {name: split.copy() for name, split in splits.items()}

    if scaler is None:
        return scaled, None

    scaler.fit(scaled["train"][feature_columns].to_numpy(dtype=float))

    for name, split in scaled.items():
        split.loc[:, feature_columns] = scaler.transform(
            split[feature_columns].to_numpy(dtype=float)
        )
        scaled[name] = split

    return scaled, scaler


def make_regression_arrays(
    splits: dict[str, pd.DataFrame],
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
    scaler_name: str = "standard",
) -> tuple[RegressionArrays, object | None]:
    """Return train/validation/test arrays for one continuous volatility target."""
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    if target_column not in DEFAULT_TARGET_COLUMNS:
        raise ValueError(
            f"Unexpected target_column={target_column!r}. "
            f"Expected one of {DEFAULT_TARGET_COLUMNS}."
        )

    cleaned = {
        name: drop_nonfinite_model_rows(
            split,
            feature_columns=feature_columns,
            target_columns=[target_column],
        )
        for name, split in splits.items()
    }
    scaled, scaler = scale_splits_train_only(
        cleaned,
        feature_columns=feature_columns,
        scaler_name=scaler_name,
    )

    arrays = RegressionArrays(
        X_train=scaled["train"][feature_columns].to_numpy(dtype=float),
        y_train=scaled["train"][target_column].to_numpy(dtype=float),
        X_val=scaled["val"][feature_columns].to_numpy(dtype=float),
        y_val=scaled["val"][target_column].to_numpy(dtype=float),
        X_test=scaled["test"][feature_columns].to_numpy(dtype=float),
        y_test=scaled["test"][target_column].to_numpy(dtype=float),
        feature_columns=list(feature_columns),
        target_column=target_column,
        scaler_name=scaler_name,
    )
    return arrays, scaler


def make_sequence_arrays(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Convert a single split DataFrame into sequence arrays.

    X[t] contains the previous `lookback` rows ending at t.
    y[t] is the target on that same final row. The target itself is already
    forward-looking in the processed dataset, so this remains leakage-safe.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    validate_columns(df, feature_columns=feature_columns, target_columns=[target_column])

    values = df[feature_columns].to_numpy(dtype=float)
    targets = df[target_column].to_numpy(dtype=float)
    dates = df["date"].reset_index(drop=True)

    X, y, y_dates = [], [], []
    for end_idx in range(lookback - 1, len(df)):
        start_idx = end_idx - lookback + 1
        X.append(values[start_idx : end_idx + 1])
        y.append(targets[end_idx])
        y_dates.append(dates.iloc[end_idx])

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), pd.Series(y_dates)
