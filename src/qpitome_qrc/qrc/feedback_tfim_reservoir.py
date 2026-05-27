from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.metrics import VolatilityForecastMetrics, evaluate_volatility_forecast
from qpitome_qrc.qrc.tfim_reservoir import (
    _apply_single_qubit_gate,
    _apply_zz_phase,
    diagnose_reservoir_feature_splits,
    expectation_x,
    expectation_z,
    expectation_zz,
    initialize_zero_state,
    make_qrc_sequence_splits,
    rx,
    ry,
    rz,
)

ObservableMode = Literal["z", "zx", "zxzz"]
TargetTransform = Literal["log", "none"]
TemporalPolicy = Literal["even", "recent", "all"]


@dataclass(frozen=True)
class FeedbackTFIMQRCConfig:
    """Sequential dense-TFIM QRC with deterministic expectation feedback.

    This v2 prototype preserves the essential feedback-reservoir mechanism:
    measured reservoir summaries at time t alter the quantum dynamics at time
    t+1. It intentionally avoids stochastic weak-measurement trajectories in the
    first implementation so feedback can be tested cleanly against no-feedback.
    """

    qubits: int = 6
    pca_components: int = 6
    lookback_days: int = 40
    temporal_steps: int = 20
    temporal_policy: TemporalPolicy = "even"
    input_qubits: tuple[int, ...] = (0, 1)
    memory_qubits: tuple[int, ...] = (2, 3, 4)
    readout_qubits: tuple[int, ...] = (5,)
    observable_mode: ObservableMode = "zxzz"
    trotter_steps_per_time: int = 1
    evolution_time: float = 0.25
    input_scale: float = np.pi / 2
    transverse_field: float = 0.5
    input_memory_coupling_scale: float = 1.2
    memory_coupling_scale: float = 1.0
    readout_coupling_scale: float = 0.7
    weak_background_coupling_scale: float = 0.15
    feedback_gain: float = 0.0
    feedback_rotation: Literal["rz", "rx"] = "rz"
    ridge_alpha: float = 3000.0
    target_transform: TargetTransform = "log"
    seed: int = 42
    disorder_strength: float = 0.10


@dataclass
class FeedbackTFIMQRCResult:
    config: FeedbackTFIMQRCConfig
    target: str
    train_metrics: VolatilityForecastMetrics
    val_metrics: VolatilityForecastMetrics
    test_metrics: VolatilityForecastMetrics
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    train_features: np.ndarray
    val_features: np.ndarray
    test_features: np.ndarray
    readout: Ridge
    feature_scaler: StandardScaler


def select_temporal_indices(
    lookback_days: int,
    temporal_steps: int,
    policy: TemporalPolicy = "even",
) -> np.ndarray:
    """Select temporal injection indices from a rolling window."""
    if temporal_steps < 1:
        raise ValueError("temporal_steps must be >= 1")
    if policy == "all":
        return np.arange(lookback_days)
    if temporal_steps > lookback_days:
        raise ValueError("temporal_steps must be <= lookback_days unless policy='all'")
    if policy == "even":
        return np.linspace(0, lookback_days - 1, temporal_steps).round().astype(int)
    if policy == "recent":
        grid = np.geomspace(1, lookback_days, temporal_steps)
        idx = lookback_days - np.round(grid).astype(int)
        return np.unique(np.clip(idx, 0, lookback_days - 1))
    raise ValueError(f"Unknown temporal policy: {policy}")


def _validate_qubit_partition(config: FeedbackTFIMQRCConfig) -> None:
    groups = [config.input_qubits, config.memory_qubits, config.readout_qubits]
    flat = [q for group in groups for q in group]
    if sorted(flat) != sorted(set(flat)):
        raise ValueError("input, memory, and readout qubits must be disjoint")
    if any(q < 0 or q >= config.qubits for q in flat):
        raise ValueError("qubit indices must be within [0, qubits)")
    if len(config.input_qubits) == 0:
        raise ValueError("at least one input qubit is required")
    if len(config.memory_qubits) == 0:
        raise ValueError("at least one memory qubit is required")
    if len(config.readout_qubits) == 0:
        raise ValueError("at least one readout qubit is required")


def _dense_coupling_matrix(config: FeedbackTFIMQRCConfig) -> np.ndarray:
    """Construct a fixed block-structured dense ZZ coupling matrix."""
    _validate_qubit_partition(config)
    n = config.qubits
    J = np.full((n, n), config.weak_background_coupling_scale, dtype=float)
    np.fill_diagonal(J, 0.0)

    input_set = set(config.input_qubits)
    memory_set = set(config.memory_qubits)
    readout_set = set(config.readout_qubits)

    for i in range(n):
        for j in range(i + 1, n):
            pair = {i, j}
            if i in memory_set and j in memory_set:
                scale = config.memory_coupling_scale
            elif (i in input_set and j in memory_set) or (j in input_set and i in memory_set):
                scale = config.input_memory_coupling_scale
            elif (i in readout_set and j in memory_set) or (j in readout_set and i in memory_set):
                scale = config.readout_coupling_scale
            elif pair & readout_set:
                scale = config.readout_coupling_scale * 0.5
            else:
                scale = config.weak_background_coupling_scale
            J[i, j] = J[j, i] = scale

    if config.disorder_strength != 0.0:
        rng = np.random.default_rng(config.seed)
        disorder = 1.0 + config.disorder_strength * rng.normal(size=(n, n))
        disorder = np.triu(disorder, k=1)
        disorder = disorder + disorder.T
        J = J * np.clip(disorder, 0.05, None)

    return J


def _field_factors(config: FeedbackTFIMQRCConfig) -> np.ndarray:
    if config.disorder_strength == 0.0:
        return np.ones(config.qubits)
    rng = np.random.default_rng(config.seed + 17)
    factors = 1.0 + config.disorder_strength * rng.normal(size=config.qubits)
    return np.clip(factors, 0.05, None)


def _encode_input_on_input_qubits(
    state: np.ndarray,
    x: np.ndarray,
    config: FeedbackTFIMQRCConfig,
) -> np.ndarray:
    clipped = np.clip(x, -3.0, 3.0) / 3.0
    angles = config.input_scale * clipped
    input_qubits = config.input_qubits

    for j, theta in enumerate(angles):
        q = input_qubits[j % len(input_qubits)]
        state = _apply_single_qubit_gate(state, ry(theta), q, config.qubits)
    return state


def _readout_feedback_signal(state: np.ndarray, config: FeedbackTFIMQRCConfig) -> float:
    values = [expectation_z(state, q, config.qubits) for q in config.readout_qubits]
    return float(np.mean(values))


def _apply_feedback(state: np.ndarray, signal: float, config: FeedbackTFIMQRCConfig) -> np.ndarray:
    if config.feedback_gain == 0.0:
        return state

    theta = config.feedback_gain * signal
    for q in config.memory_qubits:
        if config.feedback_rotation == "rz":
            gate = rz(theta)
        elif config.feedback_rotation == "rx":
            gate = rx(theta)
        else:
            raise ValueError(f"Unknown feedback_rotation: {config.feedback_rotation}")
        state = _apply_single_qubit_gate(state, gate, q, config.qubits)
    return state


def _evolve_dense_tfim_step(
    state: np.ndarray,
    config: FeedbackTFIMQRCConfig,
    J: np.ndarray,
    field_factors: np.ndarray,
) -> np.ndarray:
    n = config.qubits
    dt = config.evolution_time / max(config.trotter_steps_per_time, 1)

    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                state = _apply_zz_phase(state, i, j, J[i, j] * dt, n)

    for q in range(n):
        gate = rx(2.0 * config.transverse_field * field_factors[q] * dt)
        state = _apply_single_qubit_gate(state, gate, q, n)

    return state


def feedback_observable_features(state: np.ndarray, config: FeedbackTFIMQRCConfig) -> np.ndarray:
    """Extract observables, prioritizing memory/readout qubits but including all qubits."""
    n = config.qubits
    feats: list[float] = []

    feats.extend(expectation_z(state, q, n) for q in range(n))

    if config.observable_mode in {"zx", "zxzz"}:
        feats.extend(expectation_x(state, q, n) for q in range(n))

    if config.observable_mode == "zxzz":
        for i in range(n):
            for j in range(i + 1, n):
                feats.append(expectation_zz(state, i, j, n))

    # Explicit low-dimensional feedback/memory summaries help interpretability.
    feats.append(_readout_feedback_signal(state, config))
    feats.append(float(np.mean([expectation_z(state, q, n) for q in config.memory_qubits])))

    return np.asarray(feats, dtype=float)


def run_feedback_tfim_reservoir_for_window(
    window: np.ndarray,
    config: FeedbackTFIMQRCConfig,
) -> np.ndarray:
    """Run one sequence window through sequential dense-TFIM feedback QRC."""
    if window.ndim != 2:
        raise ValueError(f"Expected window shape (lookback, features), got {window.shape}")

    state = initialize_zero_state(config.qubits)
    temporal_indices = select_temporal_indices(
        config.lookback_days,
        config.temporal_steps,
        config.temporal_policy,
    )
    J = _dense_coupling_matrix(config)
    field_factors = _field_factors(config)

    feature_blocks: list[np.ndarray] = []
    feedback_signal = 0.0

    for idx in temporal_indices:
        state = _apply_feedback(state, feedback_signal, config)
        state = _encode_input_on_input_qubits(state, window[idx], config)

        for _ in range(config.trotter_steps_per_time):
            state = _evolve_dense_tfim_step(state, config, J, field_factors)

        feedback_signal = _readout_feedback_signal(state, config)
        feature_blocks.append(feedback_observable_features(state, config))

    return np.concatenate(feature_blocks)


def build_feedback_qrc_feature_matrix(
    X_windows: np.ndarray,
    config: FeedbackTFIMQRCConfig,
    *,
    verbose: bool = False,
) -> np.ndarray:
    rows = []
    for i, window in enumerate(X_windows):
        if verbose and i % 250 == 0:
            print(f"Feedback-QRC sample {i}/{len(X_windows)}")
        rows.append(run_feedback_tfim_reservoir_for_window(window, config))
    return np.asarray(rows, dtype=float)


def fit_feedback_qrc_readout(
    H_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: FeedbackTFIMQRCConfig,
) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    H_train_scaled = scaler.fit_transform(H_train)
    readout = Ridge(alpha=config.ridge_alpha)

    if config.target_transform == "log":
        y_fit = np.log(np.maximum(y_train, 1e-8))
    elif config.target_transform == "none":
        y_fit = y_train
    else:
        raise ValueError(f"Unknown target_transform: {config.target_transform}")

    readout.fit(H_train_scaled, y_fit)
    return readout, scaler


def predict_feedback_qrc_readout(
    readout: Ridge,
    scaler: StandardScaler,
    H: np.ndarray,
    *,
    config: FeedbackTFIMQRCConfig,
) -> np.ndarray:
    raw = readout.predict(scaler.transform(H))
    if config.target_transform == "log":
        return np.exp(raw)
    return np.maximum(raw, 1e-8)


def fit_feedback_tfim_qrc_regressor(
    sequence_splits: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]],
    *,
    config: FeedbackTFIMQRCConfig,
    target: str,
    verbose: bool = False,
) -> FeedbackTFIMQRCResult:
    X_train, y_train, _ = sequence_splits["train"]
    X_val, y_val, _ = sequence_splits["val"]
    X_test, y_test, _ = sequence_splits["test"]

    H_train = build_feedback_qrc_feature_matrix(X_train, config, verbose=verbose)
    H_val = build_feedback_qrc_feature_matrix(X_val, config, verbose=verbose)
    H_test = build_feedback_qrc_feature_matrix(X_test, config, verbose=verbose)

    readout, scaler = fit_feedback_qrc_readout(H_train, y_train, config=config)

    train_pred = predict_feedback_qrc_readout(readout, scaler, H_train, config=config)
    val_pred = predict_feedback_qrc_readout(readout, scaler, H_val, config=config)
    test_pred = predict_feedback_qrc_readout(readout, scaler, H_test, config=config)

    return FeedbackTFIMQRCResult(
        config=config,
        target=target,
        train_metrics=evaluate_volatility_forecast(y_train, train_pred),
        val_metrics=evaluate_volatility_forecast(y_val, val_pred),
        test_metrics=evaluate_volatility_forecast(y_test, test_pred),
        train_predictions=train_pred,
        val_predictions=val_pred,
        test_predictions=test_pred,
        train_features=H_train,
        val_features=H_val,
        test_features=H_test,
        readout=readout,
        feature_scaler=scaler,
    )


def summarize_feedback_qrc_result(result: FeedbackTFIMQRCResult) -> dict:
    row = asdict(result.config)
    row.update(
        {
            "model": "feedback_tfim_qrc_exact",
            "target": result.target,
            "train_n": len(result.train_predictions),
            "val_n": len(result.val_predictions),
            "test_n": len(result.test_predictions),
            "n_reservoir_features": result.train_features.shape[1],
        }
    )

    for split, metrics in [
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
    ]:
        row[f"{split}_rmse"] = metrics.rmse
        row[f"{split}_qlike"] = metrics.qlike
        row[f"{split}_mz_alpha"] = metrics.mz_alpha
        row[f"{split}_mz_beta"] = metrics.mz_beta
        row[f"{split}_mz_r2"] = metrics.mz_r2

    return row


__all__ = [
    "FeedbackTFIMQRCConfig",
    "FeedbackTFIMQRCResult",
    "build_feedback_qrc_feature_matrix",
    "diagnose_reservoir_feature_splits",
    "fit_feedback_tfim_qrc_regressor",
    "make_qrc_sequence_splits",
    "run_feedback_tfim_reservoir_for_window",
    "select_temporal_indices",
    "summarize_feedback_qrc_result",
]
