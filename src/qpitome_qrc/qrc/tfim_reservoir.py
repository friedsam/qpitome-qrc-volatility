from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ObservableMode = Literal['z', 'zx', 'zxzz']
AnchorPolicy = Literal['even', 'recent']
Topology = Literal['chain', 'full']


@dataclass(frozen=True)
class TFIMQRCConfig:
    """Exact-state TFIM reservoir configuration used by the frozen Phase 2 control."""

    qubits: int = 6
    lookback_steps: int = 12
    anchor_count: int = 10
    anchor_policy: AnchorPolicy = 'recent'
    observable_mode: ObservableMode = 'zxzz'
    trotter_steps_per_anchor: int = 3
    virtual_nodes_per_anchor: int = 3
    topology: Topology = 'full'
    coupling_scale: float = 0.7
    transverse_field: float = 0.5
    evolution_time: float = 0.5
    angle_max: float = np.pi / 2
    seed: int = 42
    collect_anchor_features: bool = True
    use_disorder: bool = True
    disorder_strength: float = 0.20


def select_anchor_indices(lookback_steps: int, anchor_count: int, policy: AnchorPolicy = 'recent') -> np.ndarray:
    if anchor_count < 1 or anchor_count > lookback_steps:
        raise ValueError('anchor_count must be between 1 and lookback_steps')
    if policy == 'even':
        return np.linspace(0, lookback_steps - 1, anchor_count).round().astype(int)
    if policy == 'recent':
        grid = np.geomspace(1, lookback_steps, anchor_count)
        idx = lookback_steps - np.round(grid).astype(int)
        return np.unique(np.clip(idx, 0, lookback_steps - 1))
    raise ValueError(f'Unknown anchor policy: {policy}')


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, tensor, axes=([1], [0]))
    updated = np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def _apply_zz_phase(state: np.ndarray, qubit_a: int, qubit_b: int, angle: float, n_qubits: int) -> np.ndarray:
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


def encode_input_angles(state: np.ndarray, x: np.ndarray, *, n_qubits: int, angle_max: float) -> np.ndarray:
    clipped = np.clip(x, -3.0, 3.0) / 3.0
    angles = angle_max * clipped
    for j, theta in enumerate(angles):
        state = _apply_single_qubit_gate(state, ry(theta), j % n_qubits, n_qubits)
    return state


def _edge_count(config: TFIMQRCConfig) -> int:
    if config.topology == 'chain':
        return max(config.qubits - 1, 0)
    if config.topology == 'full':
        return config.qubits * (config.qubits - 1) // 2
    raise ValueError(f'Unknown topology: {config.topology}')


def _fixed_disorder_factors(config: TFIMQRCConfig) -> tuple[np.ndarray, np.ndarray]:
    if not config.use_disorder or config.disorder_strength == 0.0:
        return np.ones(_edge_count(config)), np.ones(config.qubits)
    rng = np.random.default_rng(config.seed)
    edges = 1.0 + config.disorder_strength * rng.normal(size=_edge_count(config))
    fields = 1.0 + config.disorder_strength * rng.normal(size=config.qubits)
    return np.clip(edges, 0.05, None), np.clip(fields, 0.05, None)


def evolve_tfim_step(
    state: np.ndarray,
    config: TFIMQRCConfig,
    *,
    edge_factors: np.ndarray,
    field_factors: np.ndarray,
) -> np.ndarray:
    n = config.qubits
    dt = config.evolution_time / config.trotter_steps_per_anchor
    if config.topology == 'chain':
        for q in range(n - 1):
            state = _apply_zz_phase(state, q, q + 1, config.coupling_scale * edge_factors[q] * dt, n)
    elif config.topology == 'full':
        pair_idx = 0
        for qa in range(n - 1):
            for qb in range(qa + 1, n):
                state = _apply_zz_phase(state, qa, qb, config.coupling_scale * edge_factors[pair_idx] * dt, n)
                pair_idx += 1
    else:
        raise ValueError(f'Unknown topology: {config.topology}')
    for q in range(n):
        state = _apply_single_qubit_gate(
            state,
            rx(2.0 * config.transverse_field * field_factors[q] * dt),
            q,
            n,
        )
    return state


def expectation_z(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    probs = np.abs(state) ** 2
    indices = np.arange(state.size)
    bit = (indices >> (n_qubits - 1 - qubit)) & 1
    return float(np.sum(probs * (1 - 2 * bit)).real)


def expectation_x(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    indices = np.arange(state.size)
    flip_mask = 1 << (n_qubits - 1 - qubit)
    return float(np.vdot(state, state[indices ^ flip_mask]).real)


def expectation_zz(state: np.ndarray, qubit_a: int, qubit_b: int, n_qubits: int) -> float:
    probs = np.abs(state) ** 2
    indices = np.arange(state.size)
    bit_a = (indices >> (n_qubits - 1 - qubit_a)) & 1
    bit_b = (indices >> (n_qubits - 1 - qubit_b)) & 1
    return float(np.sum(probs * (1 - 2 * bit_a) * (1 - 2 * bit_b)).real)


def observable_features(state: np.ndarray, config: TFIMQRCConfig) -> np.ndarray:
    n = config.qubits
    feats: list[float] = [expectation_z(state, q, n) for q in range(n)]
    if config.observable_mode in {'zx', 'zxzz'}:
        feats.extend(expectation_x(state, q, n) for q in range(n))
    if config.observable_mode == 'zxzz':
        feats.extend(expectation_zz(state, q, q + 1, n) for q in range(n - 1))
    return np.asarray(feats, dtype=float)


def run_tfim_reservoir_for_window(window: np.ndarray, config: TFIMQRCConfig) -> np.ndarray:
    if window.shape[0] != config.lookback_steps:
        raise ValueError(f'Expected {config.lookback_steps} rows, got {window.shape}')
    if config.virtual_nodes_per_anchor > config.trotter_steps_per_anchor:
        raise ValueError('virtual_nodes_per_anchor cannot exceed trotter_steps_per_anchor')

    state = initialize_zero_state(config.qubits)
    anchors = select_anchor_indices(config.lookback_steps, config.anchor_count, config.anchor_policy)
    edge_factors, field_factors = _fixed_disorder_factors(config)
    readout_steps = set(
        np.linspace(1, config.trotter_steps_per_anchor, config.virtual_nodes_per_anchor).round().astype(int)
    )
    out: list[np.ndarray] = []

    for idx in anchors:
        state = encode_input_angles(state, window[idx], n_qubits=config.qubits, angle_max=config.angle_max)
        for step in range(1, config.trotter_steps_per_anchor + 1):
            state = evolve_tfim_step(
                state,
                config,
                edge_factors=edge_factors,
                field_factors=field_factors,
            )
            if config.collect_anchor_features and step in readout_steps:
                out.append(observable_features(state, config))

    return np.concatenate(out) if config.collect_anchor_features else observable_features(state, config)


def build_qrc_feature_matrix(X_windows: np.ndarray, config: TFIMQRCConfig, *, verbose: bool = False) -> np.ndarray:
    rows = []
    for i, window in enumerate(X_windows):
        if verbose and i % 100 == 0:
            print(f'QRC sample {i}/{len(X_windows)}')
        rows.append(run_tfim_reservoir_for_window(window, config))
    return np.asarray(rows, dtype=float)


def safe_feature_target_correlations(H: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Hc = H - H.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.linalg.norm(Hc, axis=0) * max(np.linalg.norm(yc), eps)
    corr = np.zeros(H.shape[1], dtype=float)
    valid = denom > eps
    corr[valid] = (Hc[:, valid].T @ yc) / denom[valid]
    return corr
