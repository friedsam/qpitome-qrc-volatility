from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from reservoirpy.nodes import Reservoir
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.metrics import (
    ClassificationMetrics,
    evaluate_binary_classifier,
)


COMPACT_FEATURES = [
    "spy_log_return",
    "spy_abs_return",
    "spy_range",
    "spy_dollar_volume",
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "spy_drawdown_20d",
    "vix_close",
    "vix_change",
    "vix_pct_change",
    "vix_ma_5d",
]

EXPANDED_FEATURES = [
    *COMPACT_FEATURES,
    "rv_60d",
    "rv_5d_ewm_5",
    "rv_5d_ewm_20",
    "rv_ratio_5_20",
    "rv_ratio_20_60",
    "rv_excess_5_vs_20",
    "rv_excess_5_vs_60",
]

# Default stays compact because expanded features did not improve validation PR-AUC.
DEFAULT_FEATURES = COMPACT_FEATURES

TARGET = "future_high_vol_label"

PoolingMode = Literal["final", "mean", "final_mean", "final_mean_max_std"]


@dataclass(frozen=True)
class ESNConfig:
    units: int = 300
    spectral_radius: float = 0.9
    leak_rate: float = 0.5
    input_scaling: float = 0.5
    input_connectivity: float = 0.5
    reservoir_connectivity: float = 0.1
    readout_C: float = 1.0
    seed: int = 42

    # Conservative defaults: match the earlier better-performing setup.
    washout: int = 0
    pooling: PoolingMode = "final"
    scale_states: bool = False


@dataclass
class ESNRunResult:
    config: ESNConfig
    reservoir: Reservoir
    readout: LogisticRegression
    input_scaler: StandardScaler
    state_scaler: StandardScaler | None
    threshold: float
    val_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    val_scores: np.ndarray
    test_scores: np.ndarray


def scale_sequence_splits(splits: dict) -> tuple[dict, StandardScaler]:
    """Fit input scaler on train sequences only; transform train/val/test."""
    X_train, _, _ = splits["train"]
    _, _, input_dim = X_train.shape

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, input_dim))

    scaled = {}
    for name, (X, y, dates) in splits.items():
        X_scaled = scaler.transform(X.reshape(-1, input_dim)).reshape(X.shape)
        scaled[name] = (X_scaled, y, dates)

    return scaled, scaler


def build_reservoir(config: ESNConfig, input_dim: int) -> Reservoir:
    """Build fixed ReservoirPy reservoir."""
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


def _run_reservoir_sequence(reservoir: Reservoir, seq: np.ndarray) -> np.ndarray:
    """
    Run one sequence through the reservoir with explicit reset.

    ReservoirPy versions differ in reset handling. Prefer run(..., reset=True);
    otherwise reset only if state exists.
    """
    try:
        return np.asarray(reservoir.run(seq, reset=True))
    except TypeError:
        if hasattr(reservoir, "state"):
            reservoir.reset()
        return np.asarray(reservoir.run(seq))


def reservoir_sequence_features(
    reservoir: Reservoir,
    X: np.ndarray,
    washout: int = 0,
    pooling: PoolingMode = "final",
) -> np.ndarray:
    """
    Convert sequences into reservoir features.

    X shape:
        (n_samples, seq_len, n_features)

    Pooling:
        final              -> final reservoir state
        mean               -> mean state after washout
        final_mean         -> concat(final, mean)
        final_mean_max_std -> concat(final, mean, max, std)
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
                [
                    usable[-1],
                    usable.mean(axis=0),
                    usable.max(axis=0),
                    usable.std(axis=0),
                ]
            )
        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")

        features.append(feat)

    return np.asarray(features)


def maybe_scale_state_features(
    H_train: np.ndarray,
    H_val: np.ndarray,
    H_test: np.ndarray,
    scale_states: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """Optionally scale reservoir features after reservoir transformation."""
    if not scale_states:
        return H_train, H_val, H_test, None

    scaler = StandardScaler()
    H_train_scaled = scaler.fit_transform(H_train)
    H_val_scaled = scaler.transform(H_val)
    H_test_scaled = scaler.transform(H_test)

    return H_train_scaled, H_val_scaled, H_test_scaled, scaler


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    """Choose threshold maximizing validation F1 for class 1."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def fit_single_esn(
    splits: dict,
    config: ESNConfig,
    tune_threshold: bool = True,
) -> ESNRunResult:
    """Fit one ESN config and evaluate validation/test."""
    scaled, input_scaler = scale_sequence_splits(splits)

    X_train, y_train, _ = scaled["train"]
    X_val, y_val, _ = scaled["val"]
    X_test, y_test, _ = scaled["test"]

    reservoir = build_reservoir(config, input_dim=X_train.shape[-1])

    H_train = reservoir_sequence_features(
        reservoir=reservoir,
        X=X_train,
        washout=config.washout,
        pooling=config.pooling,
    )
    H_val = reservoir_sequence_features(
        reservoir=reservoir,
        X=X_val,
        washout=config.washout,
        pooling=config.pooling,
    )
    H_test = reservoir_sequence_features(
        reservoir=reservoir,
        X=X_test,
        washout=config.washout,
        pooling=config.pooling,
    )

    H_train, H_val, H_test, state_scaler = maybe_scale_state_features(
        H_train,
        H_val,
        H_test,
        scale_states=config.scale_states,
    )

    readout = LogisticRegression(
        C=config.readout_C,
        class_weight="balanced",
        max_iter=3000,
        random_state=config.seed,
    )
    readout.fit(H_train, y_train)

    val_scores = readout.predict_proba(H_val)[:, 1]
    test_scores = readout.predict_proba(H_test)[:, 1]

    threshold = find_best_threshold(y_val, val_scores)[0] if tune_threshold else 0.5

    val_metrics = evaluate_binary_classifier(y_val, val_scores, threshold=threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_scores, threshold=threshold)

    return ESNRunResult(
        config=config,
        reservoir=reservoir,
        readout=readout,
        input_scaler=input_scaler,
        state_scaler=state_scaler,
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_scores=val_scores,
        test_scores=test_scores,
    )


def grid_configs(
    units=(300, 600, 1000),
    spectral_radius=(0.5, 0.7, 0.9),
    leak_rate=(0.2, 0.5, 0.8),
    reservoir_connectivity=(0.05, 0.1),
    readout_C=(0.1, 1.0, 10.0),
    seeds=(1, 2, 3),
    washout=(0,),
    pooling: tuple[PoolingMode, ...] = ("final",),
    scale_states=(False,),
    input_scaling=0.5,
    input_connectivity=0.5,
) -> list[ESNConfig]:
    """Generate controlled ESN search grid."""
    configs = []

    for values in product(
        units,
        spectral_radius,
        leak_rate,
        reservoir_connectivity,
        readout_C,
        seeds,
        washout,
        pooling,
        scale_states,
    ):
        units_i, sr, leak, conn, C, seed, washout_i, pooling_i, scale_i = values

        configs.append(
            ESNConfig(
                units=units_i,
                spectral_radius=sr,
                leak_rate=leak,
                input_scaling=input_scaling,
                input_connectivity=input_connectivity,
                reservoir_connectivity=conn,
                readout_C=C,
                seed=seed,
                washout=washout_i,
                pooling=pooling_i,
                scale_states=scale_i,
            )
        )

    return configs


def summarize_run(result: ESNRunResult, include_test: bool = True) -> dict:
    """Flatten one ESN result into a metrics row."""
    row = asdict(result.config)

    row.update(
        {
            "threshold": result.threshold,
            "val_balanced_accuracy": result.val_metrics.balanced_accuracy,
            "val_roc_auc": result.val_metrics.roc_auc,
            "val_pr_auc": result.val_metrics.pr_auc,
            "val_precision_class_1": result.val_metrics.precision_class_1,
            "val_recall_class_1": result.val_metrics.recall_class_1,
            "val_f1_class_1": result.val_metrics.f1_class_1,
        }
    )

    if include_test:
        row.update(
            {
                "test_balanced_accuracy": result.test_metrics.balanced_accuracy,
                "test_roc_auc": result.test_metrics.roc_auc,
                "test_pr_auc": result.test_metrics.pr_auc,
                "test_precision_class_1": result.test_metrics.precision_class_1,
                "test_recall_class_1": result.test_metrics.recall_class_1,
                "test_f1_class_1": result.test_metrics.f1_class_1,
            }
        )

    return row


def config_key(config: ESNConfig) -> tuple:
    """Hyperparameter key excluding seed."""
    d = asdict(config)
    d.pop("seed")
    return tuple(sorted(d.items()))


def run_esn_grid_search(
    splits: dict,
    configs: list[ESNConfig],
    selection_metric: str = "val_pr_auc",
    aggregate_seeds: bool = True,
    include_test_in_summary: bool = True,
) -> tuple[pd.DataFrame, ESNRunResult]:
    """
    Run controlled ESN grid.

    Selection is based on validation metrics only. If aggregate_seeds=True,
    select the hyperparameter group with best mean validation metric across seeds,
    then return the best seed-level result inside that selected group.
    """
    rows = []
    results = []

    for i, config in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {config}")
        result = fit_single_esn(splits=splits, config=config, tune_threshold=True)

        results.append(result)

        row = summarize_run(result, include_test=include_test_in_summary)
        row["_result_idx"] = len(results) - 1
        row["_config_key"] = config_key(config)
        rows.append(row)

    summary = pd.DataFrame(rows)

    if not aggregate_seeds:
        sorted_summary = summary.sort_values(selection_metric, ascending=False)
        best_result = results[int(sorted_summary.iloc[0]["_result_idx"])]

        return (
            sorted_summary.drop(columns=["_config_key"]).reset_index(drop=True),
            best_result,
        )

    grouped = (
        summary.groupby("_config_key", as_index=False)
        .agg(
            mean_selection_metric=(selection_metric, "mean"),
            std_selection_metric=(selection_metric, "std"),
            n_seeds=(selection_metric, "size"),
        )
        .sort_values("mean_selection_metric", ascending=False)
    )

    best_key = grouped.iloc[0]["_config_key"]

    candidate_rows = summary[summary["_config_key"] == best_key]
    best_row = candidate_rows.sort_values(selection_metric, ascending=False).iloc[0]
    best_result = results[int(best_row["_result_idx"])]

    display_summary = (
        summary.merge(grouped, on="_config_key", how="left")
        .sort_values(["mean_selection_metric", selection_metric], ascending=False)
        .drop(columns=["_config_key"])
        .reset_index(drop=True)
    )

    return display_summary, best_result


def prediction_frame(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build dataframe for diagnostic plots."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "y_true": y_true.astype(int),
            "y_score": y_score,
            "y_pred": (y_score >= threshold).astype(int),
        }
    )