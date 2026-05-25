from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.metrics import VolatilityForecastMetrics, evaluate_volatility_forecast

try:
    from reservoirpy.nodes import Reservoir
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    Reservoir = None
    _RESERVOIRPY_IMPORT_ERROR = exc
else:
    _RESERVOIRPY_IMPORT_ERROR = None


PoolingMode = Literal["final", "mean", "final_mean", "final_mean_max_std"]


@dataclass(frozen=True)
class ESNRegressionConfig:
    """Fixed ReservoirPy ESN configuration for Phase 2 regression baselines."""

    units: int = 300
    spectral_radius: float = 0.7
    leak_rate: float = 0.5
    input_scaling: float = 0.5
    input_connectivity: float = 0.5
    reservoir_connectivity: float = 0.1
    ridge_alpha: float = 1.0
    seed: int = 42
    washout: int = 0
    pooling: PoolingMode = "final"
    scale_states: bool = False


@dataclass
class ESNRegressionRunResult:
    config: ESNRegressionConfig
    target: str
    feature_set: str
    seq_len: int
    reservoir: object
    readout: Ridge
    input_scaler: StandardScaler
    state_scaler: StandardScaler | None
    train_metrics: VolatilityForecastMetrics
    val_metrics: VolatilityForecastMetrics
    test_metrics: VolatilityForecastMetrics
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    train_dates: pd.Series
    val_dates: pd.Series
    test_dates: pd.Series


def require_reservoirpy() -> None:
    """Raise a useful error if ReservoirPy is unavailable."""
    if Reservoir is None:
        raise ImportError(
            "ReservoirPy is required for ESN regression baselines. "
            "Install with `python -m pip install reservoirpy`."
        ) from _RESERVOIRPY_IMPORT_ERROR


def scale_sequence_splits(
    sequence_splits: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, pd.Series]], StandardScaler]:
    """Fit input scaler on train sequences only; transform train/val/test."""
    X_train, _, _ = sequence_splits["train"]
    _, _, input_dim = X_train.shape

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, input_dim))

    scaled = {}
    for name, (X, y, dates) in sequence_splits.items():
        X_scaled = scaler.transform(X.reshape(-1, input_dim)).reshape(X.shape)
        scaled[name] = (X_scaled, y, dates)

    return scaled, scaler


def build_reservoir(config: ESNRegressionConfig, input_dim: int):
    """Build a ReservoirPy reservoir with explicit Phase 2 config fields."""
    require_reservoirpy()
    return Reservoir(
        units=config.units,
        lr=config.leak_rate,
        sr=config.spectral_radius,
        input_scaling=config.input_scaling,
        input_connectivity=config.input_connectivity,
        rc_connectivity=config.reservoir_connectivity,
        input_dim=input_dim,
        seed=config.seed,
    )


def _run_reservoir_sequence(reservoir, seq: np.ndarray) -> np.ndarray:
    """Run one sequence through the reservoir with explicit reset handling."""
    try:
        return np.asarray(reservoir.run(seq, reset=True))
    except TypeError:
        if hasattr(reservoir, "state"):
            reservoir.reset()
        return np.asarray(reservoir.run(seq))


def reservoir_sequence_features(
    reservoir,
    X: np.ndarray,
    *,
    washout: int = 0,
    pooling: PoolingMode = "final",
) -> np.ndarray:
    """Convert sequence inputs into reservoir-state features.

    X shape is (n_samples, seq_len, n_features). Pooling modes intentionally
    match the earlier binary-classification ESN infrastructure, but are used
    here with a Ridge regression readout.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (samples, seq_len, features), got {X.shape}")

    seq_len = X.shape[1]
    if washout < 0:
        raise ValueError("washout must be >= 0")
    if washout >= seq_len:
        raise ValueError(f"washout={washout} must be < seq_len={seq_len}")

    features = []
    for seq in X:
        trajectory = _run_reservoir_sequence(reservoir, seq)
        if trajectory.ndim != 2:
            raise ValueError(f"Expected 2D reservoir trajectory, got {trajectory.shape}")

        usable = trajectory[washout:] if washout > 0 else trajectory

        if pooling == "final":
            feat = usable[-1]
        elif pooling == "mean":
            feat = usable.mean(axis=0)
        elif pooling == "final_mean":
            feat = np.concatenate([usable[-1], usable.mean(axis=0)])
        elif pooling == "final_mean_max_std":
            feat = np.concatenate(
                [usable[-1], usable.mean(axis=0), usable.max(axis=0), usable.std(axis=0)]
            )
        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")

        features.append(feat)

    return np.asarray(features, dtype=float)


def maybe_scale_state_features(
    H_train: np.ndarray,
    H_val: np.ndarray,
    H_test: np.ndarray,
    *,
    scale_states: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """Optionally scale reservoir-state features using train statistics only."""
    if not scale_states:
        return H_train, H_val, H_test, None

    scaler = StandardScaler()
    H_train_scaled = scaler.fit_transform(H_train)
    H_val_scaled = scaler.transform(H_val)
    H_test_scaled = scaler.transform(H_test)
    return H_train_scaled, H_val_scaled, H_test_scaled, scaler


def fit_single_esn_regressor(
    sequence_splits: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]],
    *,
    config: ESNRegressionConfig,
    target: str,
    feature_set: str,
    seq_len: int,
) -> ESNRegressionRunResult:
    """Fit one ESN regressor and evaluate train/validation/test metrics."""
    scaled, input_scaler = scale_sequence_splits(sequence_splits)

    X_train, y_train, train_dates = scaled["train"]
    X_val, y_val, val_dates = scaled["val"]
    X_test, y_test, test_dates = scaled["test"]

    reservoir = build_reservoir(config, input_dim=X_train.shape[-1])

    H_train = reservoir_sequence_features(
        reservoir,
        X_train,
        washout=config.washout,
        pooling=config.pooling,
    )
    H_val = reservoir_sequence_features(
        reservoir,
        X_val,
        washout=config.washout,
        pooling=config.pooling,
    )
    H_test = reservoir_sequence_features(
        reservoir,
        X_test,
        washout=config.washout,
        pooling=config.pooling,
    )

    H_train, H_val, H_test, state_scaler = maybe_scale_state_features(
        H_train,
        H_val,
        H_test,
        scale_states=config.scale_states,
    )

    readout = Ridge(alpha=config.ridge_alpha, random_state=config.seed)
    readout.fit(H_train, y_train)

    train_pred = np.maximum(readout.predict(H_train), 1e-8)
    val_pred = np.maximum(readout.predict(H_val), 1e-8)
    test_pred = np.maximum(readout.predict(H_test), 1e-8)

    return ESNRegressionRunResult(
        config=config,
        target=target,
        feature_set=feature_set,
        seq_len=seq_len,
        reservoir=reservoir,
        readout=readout,
        input_scaler=input_scaler,
        state_scaler=state_scaler,
        train_metrics=evaluate_volatility_forecast(y_train, train_pred),
        val_metrics=evaluate_volatility_forecast(y_val, val_pred),
        test_metrics=evaluate_volatility_forecast(y_test, test_pred),
        train_predictions=train_pred,
        val_predictions=val_pred,
        test_predictions=test_pred,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
    )


def summarize_esn_regression_run(result: ESNRegressionRunResult) -> dict:
    """Flatten one ESN regression run into a metrics row."""
    row = asdict(result.config)
    row.update(
        {
            "model": "esn_regression",
            "feature_set": result.feature_set,
            "target": result.target,
            "seq_len": result.seq_len,
            "train_n": len(result.train_predictions),
            "val_n": len(result.val_predictions),
            "test_n": len(result.test_predictions),
        }
    )

    for split_name, metrics in [
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
    ]:
        row[f"{split_name}_rmse"] = metrics.rmse
        row[f"{split_name}_qlike"] = metrics.qlike
        row[f"{split_name}_mz_alpha"] = metrics.mz_alpha
        row[f"{split_name}_mz_beta"] = metrics.mz_beta
        row[f"{split_name}_mz_r2"] = metrics.mz_r2

    return row


def aggregate_esn_regression_seeds(
    runs: pd.DataFrame,
    *,
    selection_metric: str = "val_rmse",
) -> pd.DataFrame:
    """Aggregate ESN regression rows across seeds for stable model comparison."""
    config_cols = [
        "model",
        "feature_set",
        "target",
        "seq_len",
        "units",
        "spectral_radius",
        "leak_rate",
        "input_scaling",
        "input_connectivity",
        "reservoir_connectivity",
        "ridge_alpha",
        "washout",
        "pooling",
        "scale_states",
    ]
    config_cols = [c for c in config_cols if c in runs.columns]

    agg = (
        runs.groupby(config_cols, as_index=False)
        .agg(
            mean_val_rmse=("val_rmse", "mean"),
            std_val_rmse=("val_rmse", "std"),
            mean_val_qlike=("val_qlike", "mean"),
            mean_val_mz_r2=("val_mz_r2", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_test_qlike=("test_qlike", "mean"),
            mean_test_mz_r2=("test_mz_r2", "mean"),
            n_seeds=(selection_metric, "size"),
        )
        .sort_values("mean_val_rmse", ascending=True)
        .reset_index(drop=True)
    )
    return agg
