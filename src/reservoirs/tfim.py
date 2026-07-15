"""Exact-state TFIM reservoir used by the canonical Phase 2 QRC model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ObservableMode = Literal["z", "zx", "zxzz"]
AnchorPolicy = Literal["even", "recent"]
Topology = Literal["chain", "full"]


@dataclass(frozen=True)
class TFIMQRCConfig:
    qubits: int = 6
    pca_components: int = 6
    lookback_days: int = 40
    anchor_count: int = 10
    anchor_policy: AnchorPolicy = "recent"
    observable_mode: ObservableMode = "zxzz"
    trotter_steps_per_anchor: int = 3
    virtual_nodes_per_anchor: int = 3
    topology: Topology = "full"
    coupling_scale: float = 0.7
    transverse_field: float = 0.5
    evolution_time: float = 0.5
    angle_max: float = np.pi / 2
    ridge_alpha: float = 1000.0
    target_transform: Literal["log", "none"] = "log"
    seed: int = 42
    collect_anchor_features: bool = True
    use_disorder: bool = True
    disorder_strength: float = 0.20
    input_leak: float = 0.3
    clip_lower_percentile: float = 1.0
    clip_upper_percentile: float = 99.0
    top_k_features: int = 240


def config_to_dict(config: TFIMQRCConfig) -> dict:
    return asdict(config)


def select_anchor_indices(
    lookback_days: int,
    anchor_count: int,
    policy: AnchorPolicy = "even",
) -> np.ndarray:
    if anchor_count < 1:
        raise ValueError("anchor_count must be >= 1")
    if anchor_count > lookback_days:
        raise ValueError("anchor_count must be <= lookback_days")
    if policy == "even":
        return np.linspace(0, lookback_days - 1, anchor_count).round().astype(int)
    if policy == "recent":
        grid = np.geomspace(1, lookback_days, anchor_count)
        indices = lookback_days - np.round(grid).astype(int)
        return np.unique(np.clip(indices, 0, lookback_days - 1))
    raise ValueError(f"Unknown anchor policy: {policy}")


def leaky_integrate_windows(X: np.ndarray, leak: float = 0.3) -> np.ndarray:
    out = np.zeros_like(X, dtype=float)
    for sample_index, window in enumerate(X):
        state = np.zeros(window.shape[1], dtype=float)
        for time_index, input_vector in enumerate(window):
            state = (1.0 - leak) * state + leak * input_vector
            out[sample_index, time_index] = state
    return out


def _apply_single_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, tensor, axes=([1], [0]))
    updated = np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def _apply_zz_phase(
    state: np.ndarray,
    qubit_a: int,
    qubit_b: int,
    angle: float,
    n_qubits: int,
) -> np.ndarray:
    indices = np.arange(state.size)
    bit_a = (indices >> (n_qubits - 1 - qubit_a)) & 1
    bit_b = (indices >> (n_qubits - 1 - qubit_b)) & 1
    z_a = 1 - 2 * bit_a
    z_b = 1 - 2 * bit_b
    return state * np.exp(-1j * angle * z_a * z_b)


def rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


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
    angles = angle_max * (np.clip(x, -3.0, 3.0) / 3.0)
    for index, theta in enumerate(angles):
        state = _apply_single_qubit_gate(state, ry(theta), index % n_qubits, n_qubits)
    return state


def _edge_count(config: TFIMQRCConfig) -> int:
    if config.topology == "chain":
        return max(config.qubits - 1, 0)
    if config.topology == "full":
        return config.qubits * (config.qubits - 1) // 2
    raise ValueError(f"Unknown topology: {config.topology}")


def _fixed_disorder_factors(config: TFIMQRCConfig) -> tuple[np.ndarray, np.ndarray]:
    if not config.use_disorder or config.disorder_strength == 0.0:
        return np.ones(_edge_count(config)), np.ones(config.qubits)
    rng = np.random.default_rng(config.seed)
    edge_factors = 1.0 + config.disorder_strength * rng.normal(size=_edge_count(config))
    field_factors = 1.0 + config.disorder_strength * rng.normal(size=config.qubits)
    return np.clip(edge_factors, 0.05, None), np.clip(field_factors, 0.05, None)


def evolve_tfim_step(
    state: np.ndarray,
    config: TFIMQRCConfig,
    *,
    edge_factors: np.ndarray,
    field_factors: np.ndarray,
) -> np.ndarray:
    n = config.qubits
    dt = config.evolution_time / max(config.trotter_steps_per_anchor, 1)
    if config.topology == "chain":
        for qubit in range(n - 1):
            state = _apply_zz_phase(
                state,
                qubit,
                qubit + 1,
                config.coupling_scale * edge_factors[qubit] * dt,
                n,
            )
    elif config.topology == "full":
        pair_index = 0
        for qubit_a in range(n - 1):
            for qubit_b in range(qubit_a + 1, n):
                state = _apply_zz_phase(
                    state,
                    qubit_a,
                    qubit_b,
                    config.coupling_scale * edge_factors[pair_index] * dt,
                    n,
                )
                pair_index += 1
    else:
        raise ValueError(f"Unknown topology: {config.topology}")

    for qubit in range(n):
        state = _apply_single_qubit_gate(
            state,
            rx(2.0 * config.transverse_field * field_factors[qubit] * dt),
            qubit,
            n,
        )
    return state


def expectation_z(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    probabilities = np.abs(state) ** 2
    indices = np.arange(state.size)
    bit = (indices >> (n_qubits - 1 - qubit)) & 1
    return float(np.sum(probabilities * (1 - 2 * bit)).real)


def expectation_x(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    indices = np.arange(state.size)
    flipped = state[indices ^ (1 << (n_qubits - 1 - qubit))]
    return float(np.vdot(state, flipped).real)


def expectation_zz(
    state: np.ndarray,
    qubit_a: int,
    qubit_b: int,
    n_qubits: int,
) -> float:
    probabilities = np.abs(state) ** 2
    indices = np.arange(state.size)
    bit_a = (indices >> (n_qubits - 1 - qubit_a)) & 1
    bit_b = (indices >> (n_qubits - 1 - qubit_b)) & 1
    return float(np.sum(probabilities * (1 - 2 * bit_a) * (1 - 2 * bit_b)).real)


def observable_features(state: np.ndarray, config: TFIMQRCConfig) -> np.ndarray:
    n = config.qubits
    features: list[float] = [expectation_z(state, qubit, n) for qubit in range(n)]
    if config.observable_mode in {"zx", "zxzz"}:
        features.extend(expectation_x(state, qubit, n) for qubit in range(n))
    if config.observable_mode == "zxzz":
        features.extend(expectation_zz(state, qubit, qubit + 1, n) for qubit in range(n - 1))
    return np.asarray(features, dtype=float)


def run_tfim_reservoir_for_window(window: np.ndarray, config: TFIMQRCConfig) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(f"Expected window shape (lookback, features), got {window.shape}")
    if config.virtual_nodes_per_anchor < 1:
        raise ValueError("virtual_nodes_per_anchor must be >= 1")
    if config.virtual_nodes_per_anchor > config.trotter_steps_per_anchor:
        raise ValueError("virtual_nodes_per_anchor must be <= trotter_steps_per_anchor")

    state = initialize_zero_state(config.qubits)
    anchor_indices = select_anchor_indices(
        config.lookback_days,
        config.anchor_count,
        config.anchor_policy,
    )
    edge_factors, field_factors = _fixed_disorder_factors(config)
    feature_blocks: list[np.ndarray] = []
    readout_steps = set(
        np.linspace(
            1,
            config.trotter_steps_per_anchor,
            config.virtual_nodes_per_anchor,
        ).round().astype(int)
    )

    for anchor_index in anchor_indices:
        state = encode_input_angles(
            state,
            window[anchor_index],
            n_qubits=config.qubits,
            angle_max=config.angle_max,
        )
        for step in range(1, config.trotter_steps_per_anchor + 1):
            state = evolve_tfim_step(
                state,
                config,
                edge_factors=edge_factors,
                field_factors=field_factors,
            )
            if config.collect_anchor_features and step in readout_steps:
                feature_blocks.append(observable_features(state, config))

    if config.collect_anchor_features:
        return np.concatenate(feature_blocks)
    return observable_features(state, config)


def build_qrc_feature_matrix(
    X_windows: np.ndarray,
    config: TFIMQRCConfig,
    *,
    verbose: bool = False,
) -> np.ndarray:
    rows = []
    for index, window in enumerate(X_windows):
        if verbose and index % 250 == 0:
            print(f"TFIM-QRC sample {index}/{len(X_windows)}")
        rows.append(run_tfim_reservoir_for_window(window, config))
    return np.asarray(rows, dtype=float)


def safe_feature_target_correlations(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_features = H - H.mean(axis=0, keepdims=True)
    centered_target = y - y.mean()
    denominator = np.linalg.norm(centered_features, axis=0) * max(
        np.linalg.norm(centered_target),
        1e-12,
    )
    correlations = np.zeros(H.shape[1], dtype=float)
    valid = denominator > 1e-12
    correlations[valid] = (
        centered_features[:, valid].T @ centered_target
    ) / denominator[valid]
    return correlations


def fit_canonical_readout(
    H_train_raw: np.ndarray,
    y_train: np.ndarray,
    config: TFIMQRCConfig,
) -> tuple[Ridge, StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.percentile(H_train_raw, config.clip_lower_percentile, axis=0)
    upper = np.percentile(H_train_raw, config.clip_upper_percentile, axis=0)
    clipped = np.clip(H_train_raw, lower, upper)
    correlations = safe_feature_target_correlations(clipped, y_train)
    selected = np.argsort(np.abs(correlations))[-min(config.top_k_features, H_train_raw.shape[1]):]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clipped[:, selected])
    readout = Ridge(alpha=config.ridge_alpha)
    target = np.log(np.maximum(y_train, 1e-8)) if config.target_transform == "log" else y_train
    readout.fit(scaled, target)
    return readout, scaler, selected, lower, upper


def predict_canonical_readout(
    H_raw: np.ndarray,
    *,
    readout: Ridge,
    scaler: StandardScaler,
    selected: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: TFIMQRCConfig,
) -> np.ndarray:
    features = np.clip(H_raw, lower, upper)[:, selected]
    raw = readout.predict(scaler.transform(features))
    if config.target_transform == "log":
        return np.exp(raw)
    return np.maximum(raw, 1e-8)
