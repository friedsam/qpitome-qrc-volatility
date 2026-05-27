from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.metrics import VolatilityForecastMetrics, evaluate_volatility_forecast

ObservableMode = Literal["z", "zx", "zxzz"]
AnchorPolicy = Literal["even", "recent"]


@dataclass(frozen=True)
class TFIMQRCConfig:
    """Exact-state TFIM QRC prototype configuration.

    This is intentionally simulator-first and small-qubit. It is designed for
    Phase 2/early Phase 3 architecture validation, not final performance.
    """

    qubits: int = 6
    pca_components: int = 6
    lookback_days: int = 40
    anchor_count: int = 6
    anchor_policy: AnchorPolicy = "even"
    observable_mode: ObservableMode = "z"
    trotter_steps_per_anchor: int = 1
    coupling_scale: float = 0.7
    transverse_field: float = 0.5
    evolution_time: float = 0.5
    angle_max: float = np.pi / 2
    ridge_alpha: float = 10.0
    target_transform: Literal["log", "none"] = "log"
    seed: int = 42
    collect_anchor_features: bool = False


@dataclass
class TFIMQRCResult:
    config: TFIMQRCConfig
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
    feature_scaler: StandardScaler | None


def select_anchor_indices(
    lookback_days: int,
    anchor_count: int,
    policy: AnchorPolicy = "even",
) -> np.ndarray:
    """Select temporal anchor indices from a rolling window."""
    if anchor_count < 1:
        raise ValueError("anchor_count must be >= 1")
    if anchor_count > lookback_days:
        raise ValueError("anchor_count must be <= lookback_days")

    if policy == "even":
        return np.linspace(0, lookback_days - 1, anchor_count).round().astype(int)
    if policy == "recent":
        # Coarse long-history anchors plus denser recent anchors.
        grid = np.geomspace(1, lookback_days, anchor_count)
        idx = lookback_days - np.round(grid).astype(int)
        return np.unique(np.clip(idx, 0, lookback_days - 1))
    raise ValueError(f"Unknown anchor policy: {policy}")


def make_qrc_sequence_splits(
    splits: dict[str, pd.DataFrame],
    *,
    feature_columns: list[str],
    target_column: str,
    lookback_days: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.Series]]:
    """Build split-local rolling windows for QRC inputs."""
    from qpitome_qrc.data.features import make_sequence_arrays

    return {
        name: make_sequence_arrays(
            split,
            feature_columns=feature_columns,
            target_column=target_column,
            lookback=lookback_days,
        )
        for name, split in splits.items()
    }


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a 2x2 gate to one qubit of a statevector."""
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, tensor, axes=([1], [0]))
    updated = np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def _apply_zz_phase(state: np.ndarray, qubit_a: int, qubit_b: int, angle: float, n_qubits: int) -> np.ndarray:
    """Apply exp(-i angle Z_a Z_b) to a statevector."""
    indices = np.arange(state.size)
    bit_a = (indices >> (n_qubits - 1 - qubit_a)) & 1
    bit_b = (indices >> (n_qubits - 1 - qubit_b)) & 1
    z_a = 1 - 2 * bit_a
    z_b = 1 - 2 * bit_b
    phase = np.exp(-1j * angle * z_a * z_b)
    return state * phase


def rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def initialize_zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return state


def encode_input_angles(
    state: np.ndarray,
    x: np.ndarray,
    *,
    n_qubits: int,
    angle_max: float,
) -> np.ndarray:
    """Angle-encode one PCA vector using Ry rotations.

    If the PCA dimension exceeds qubit count, components are cycled/re-uploaded.
    """
    clipped = np.clip(x, -3.0, 3.0) / 3.0
    angles = angle_max * clipped

    for j, theta in enumerate(angles):
        q = j % n_qubits
        state = _apply_single_qubit_gate(state, ry(theta), q, n_qubits)
    return state


def evolve_tfim_step(state: np.ndarray, config: TFIMQRCConfig) -> np.ndarray:
    """Apply one shallow nearest-neighbor TFIM Trotter step."""
    n = config.qubits
    dt = config.evolution_time / max(config.trotter_steps_per_anchor, 1)

    # ZZ nearest-neighbor interaction layer.
    for q in range(n - 1):
        state = _apply_zz_phase(state, q, q + 1, config.coupling_scale * dt, n)

    # Transverse-field X layer.
    x_gate = rx(2.0 * config.transverse_field * dt)
    for q in range(n):
        state = _apply_single_qubit_gate(state, x_gate, q, n)

    return state


def run_tfim_reservoir_for_window(window: np.ndarray, config: TFIMQRCConfig) -> np.ndarray:
    """Run one rolling-window sample through the TFIM reservoir.

    By default, this returns final-state observables only. If
    ``collect_anchor_features`` is enabled, observables are collected after each
    temporal anchor and concatenated. This exposes virtual-node style temporal
    readout features while preserving the May-25 final-state behavior by
    default.
    """
    if window.ndim != 2:
        raise ValueError(f"Expected window shape (lookback, features), got {window.shape}")

    state = initialize_zero_state(config.qubits)
    anchor_indices = select_anchor_indices(
        config.lookback_days,
        config.anchor_count,
        config.anchor_policy,
    )

    anchor_features: list[np.ndarray] = []

    for idx in anchor_indices:
        state = encode_input_angles(
            state,
            window[idx],
            n_qubits=config.qubits,
            angle_max=config.angle_max,
        )
        for _ in range(config.trotter_steps_per_anchor):
            state = evolve_tfim_step(state, config)

        if config.collect_anchor_features:
            anchor_features.append(observable_features(state, config))

    if config.collect_anchor_features:
        return np.concatenate(anchor_features)

    return observable_features(state, config)


def expectation_z(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    probs = np.abs(state) ** 2
    indices = np.arange(state.size)
    bit = (indices >> (n_qubits - 1 - qubit)) & 1
    z = 1 - 2 * bit
    return float(np.sum(probs * z).real)


def expectation_x(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    indices = np.arange(state.size)
    flip_mask = 1 << (n_qubits - 1 - qubit)
    flipped = state[indices ^ flip_mask]
    return float(np.vdot(state, flipped).real)


def expectation_zz(state: np.ndarray, qubit_a: int, qubit_b: int, n_qubits: int) -> float:
    probs = np.abs(state) ** 2
    indices = np.arange(state.size)
    bit_a = (indices >> (n_qubits - 1 - qubit_a)) & 1
    bit_b = (indices >> (n_qubits - 1 - qubit_b)) & 1
    z_a = 1 - 2 * bit_a
    z_b = 1 - 2 * bit_b
    return float(np.sum(probs * z_a * z_b).real)


def observable_features(state: np.ndarray, config: TFIMQRCConfig) -> np.ndarray:
    """Extract Z, X, and/or nearest-neighbor ZZ expectations."""
    n = config.qubits
    feats: list[float] = []

    # Always include Z in all modes.
    feats.extend(expectation_z(state, q, n) for q in range(n))

    if config.observable_mode in {"zx", "zxzz"}:
        feats.extend(expectation_x(state, q, n) for q in range(n))

    if config.observable_mode == "zxzz":
        feats.extend(expectation_zz(state, q, q + 1, n) for q in range(n - 1))

    return np.asarray(feats, dtype=float)


def build_qrc_feature_matrix(
    X_windows: np.ndarray,
    config: TFIMQRCConfig,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Build reservoir feature matrix for a sequence-window tensor."""
    rows = []
    for i, window in enumerate(X_windows):
        if verbose and i % 250 == 0:
            print(f"QRC sample {i}/{len(X_windows)}")
        rows.append(run_tfim_reservoir_for_window(window, config))
    return np.asarray(rows, dtype=float)


def _safe_feature_target_correlations(H: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute feature-target correlations, returning zero for degenerate columns."""
    H = np.asarray(H, dtype=float)
    y = np.asarray(y, dtype=float)

    H_centered = H - H.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()

    h_norm = np.linalg.norm(H_centered, axis=0)
    y_norm = np.linalg.norm(y_centered)
    denom = h_norm * max(y_norm, eps)

    corr = np.zeros(H.shape[1], dtype=float)
    valid = denom > eps
    corr[valid] = (H_centered[:, valid].T @ y_centered) / denom[valid]
    return corr


def diagnose_reservoir_features(
    H: np.ndarray,
    y: np.ndarray | None = None,
    *,
    name: str = "split",
    near_constant_tol: float = 1e-8,
    sv_tol: float = 1e-10,
) -> dict:
    """Return compact diagnostics for a reservoir feature matrix.

    These diagnostics are intended to decide whether QRC features are useful
    enough to justify model tuning. They should be inspected before large
    parameter sweeps.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError(f"Expected H shape (samples, features), got {H.shape}")

    std = H.std(axis=0)
    centered = H - H.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    positive_sv = singular_values[singular_values > sv_tol]

    if positive_sv.size == 0:
        effective_rank = 0.0
        condition_number = np.inf
    else:
        weights = positive_sv / positive_sv.sum()
        effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
        condition_number = float(positive_sv[0] / positive_sv[-1])

    diagnostics = {
        "split": name,
        "n_samples": int(H.shape[0]),
        "n_features": int(H.shape[1]),
        "near_constant_features": int(np.sum(std < near_constant_tol)),
        "feature_std_min": float(np.min(std)),
        "feature_std_median": float(np.median(std)),
        "feature_std_max": float(np.max(std)),
        "effective_rank": effective_rank,
        "condition_number": condition_number,
    }

    if y is not None:
        corr = _safe_feature_target_correlations(H, np.asarray(y, dtype=float))
        diagnostics.update(
            {
                "mean_abs_feature_target_corr": float(np.mean(np.abs(corr))),
                "max_abs_feature_target_corr": float(np.max(np.abs(corr))),
            }
        )

    return diagnostics


def diagnose_reservoir_feature_splits(
    H_train: np.ndarray,
    H_val: np.ndarray,
    H_test: np.ndarray,
    y_train: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> pd.DataFrame:
    """Diagnose train/validation/test reservoir feature matrices."""
    rows = [
        diagnose_reservoir_features(H_train, y_train, name="train"),
        diagnose_reservoir_features(H_val, y_val, name="val"),
        diagnose_reservoir_features(H_test, y_test, name="test"),
    ]

    train_mean = np.asarray(H_train, dtype=float).mean(axis=0)
    train_std = np.asarray(H_train, dtype=float).std(axis=0) + 1e-12

    for row, H in zip(rows, [H_train, H_val, H_test], strict=True):
        z_shift = (np.asarray(H, dtype=float).mean(axis=0) - train_mean) / train_std
        row["mean_abs_shift_vs_train"] = float(np.mean(np.abs(z_shift)))
        row["max_abs_shift_vs_train"] = float(np.max(np.abs(z_shift)))

    return pd.DataFrame(rows)


def fit_qrc_readout(
    H_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: TFIMQRCConfig,
) -> tuple[Ridge, StandardScaler | None]:
    """Fit ridge readout on QRC features."""
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


def predict_qrc_readout(
    readout: Ridge,
    scaler: StandardScaler,
    H: np.ndarray,
    *,
    config: TFIMQRCConfig,
) -> np.ndarray:
    raw = readout.predict(scaler.transform(H))
    if config.target_transform == "log":
        return np.exp(raw)
    return np.maximum(raw, 1e-8)


def fit_tfim_qrc_regressor(
    sequence_splits: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]],
    *,
    config: TFIMQRCConfig,
    target: str,
    verbose: bool = False,
) -> TFIMQRCResult:
    """End-to-end exact-state TFIM-QRC regression prototype."""
    X_train, y_train, _ = sequence_splits["train"]
    X_val, y_val, _ = sequence_splits["val"]
    X_test, y_test, _ = sequence_splits["test"]

    H_train = build_qrc_feature_matrix(X_train, config, verbose=verbose)
    H_val = build_qrc_feature_matrix(X_val, config, verbose=verbose)
    H_test = build_qrc_feature_matrix(X_test, config, verbose=verbose)

    readout, scaler = fit_qrc_readout(H_train, y_train, config=config)

    train_pred = predict_qrc_readout(readout, scaler, H_train, config=config)
    val_pred = predict_qrc_readout(readout, scaler, H_val, config=config)
    test_pred = predict_qrc_readout(readout, scaler, H_test, config=config)

    return TFIMQRCResult(
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


def summarize_qrc_result(result: TFIMQRCResult) -> dict:
    """Flatten one QRC run to a result row."""
    row = asdict(result.config)
    row.update(
        {
            "model": "tfim_qrc_exact",
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
