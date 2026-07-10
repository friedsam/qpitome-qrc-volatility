"""Small fixed TFIM reservoir for task-aligned branch-transition paths.

The purpose of this module is diagnostic, not architectural search. It implements
one deterministic four-qubit temporal reservoir matched to the frozen four-channel
transition path and exposes eight final observables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from qpitome_qrc.regimes.branch_transition_path import (
    TRANSITION_PATH_CLIP_BOUNDS,
    TRANSITION_PATH_COLUMNS,
)


@dataclass(frozen=True)
class TFIMTransitionConfig:
    n_qubits: int = 4
    coupling_j: float = 1.0
    transverse_h: float = 0.8
    dt: float = 0.35
    encoding_angle_scale: float = np.pi / 2.0


I2 = np.eye(2, dtype=complex)
X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _kron_all(operators: list[np.ndarray]) -> np.ndarray:
    out = operators[0]
    for operator in operators[1:]:
        out = np.kron(out, operator)
    return out


def _single_qubit_operator(operator: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    operators = [I2] * n_qubits
    operators[qubit] = operator
    return _kron_all(operators)


def _two_qubit_operator(
    left_operator: np.ndarray,
    left_qubit: int,
    right_operator: np.ndarray,
    right_qubit: int,
    n_qubits: int,
) -> np.ndarray:
    operators = [I2] * n_qubits
    operators[left_qubit] = left_operator
    operators[right_qubit] = right_operator
    return _kron_all(operators)


def _ry(theta: float) -> np.ndarray:
    return np.cos(theta / 2.0) * I2 - 1.0j * np.sin(theta / 2.0) * Y


def scale_transition_windows(windows: np.ndarray) -> np.ndarray:
    """Map fixed clipped task channels to [-1, 1] without fitted preprocessing."""
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected windows with shape (episode, time, channel); got {values.shape}")
    if values.shape[2] != len(TRANSITION_PATH_COLUMNS):
        raise ValueError(
            f"Expected {len(TRANSITION_PATH_COLUMNS)} channels; got {values.shape[2]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Transition windows contain non-finite values")

    out = np.empty_like(values, dtype=float)
    for channel_idx, channel in enumerate(TRANSITION_PATH_COLUMNS):
        lower, upper = TRANSITION_PATH_CLIP_BOUNDS[channel]
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        out[:, :, channel_idx] = (values[:, :, channel_idx] - midpoint) / half_range
    return np.clip(out, -1.0, 1.0)


def make_tfim_evolution(config: TFIMTransitionConfig | None = None) -> np.ndarray:
    """Return one fixed ring-TFIM evolution operator."""
    cfg = config or TFIMTransitionConfig()
    if cfg.n_qubits != 4:
        raise ValueError("Task-aligned TFIM control is fixed to four qubits")

    dim = 2 ** cfg.n_qubits
    hamiltonian = np.zeros((dim, dim), dtype=complex)

    for qubit in range(cfg.n_qubits):
        neighbor = (qubit + 1) % cfg.n_qubits
        hamiltonian += cfg.coupling_j * _two_qubit_operator(
            Z,
            qubit,
            Z,
            neighbor,
            cfg.n_qubits,
        )
        hamiltonian += cfg.transverse_h * _single_qubit_operator(
            X,
            qubit,
            cfg.n_qubits,
        )

    return expm(-1.0j * cfg.dt * hamiltonian)


def make_observables(config: TFIMTransitionConfig | None = None) -> tuple[np.ndarray, ...]:
    """Return 4 local-Z and 4 nearest-neighbor ring-ZZ observables."""
    cfg = config or TFIMTransitionConfig()
    local = tuple(
        _single_qubit_operator(Z, qubit, cfg.n_qubits)
        for qubit in range(cfg.n_qubits)
    )
    pair = tuple(
        _two_qubit_operator(
            Z,
            qubit,
            Z,
            (qubit + 1) % cfg.n_qubits,
            cfg.n_qubits,
        )
        for qubit in range(cfg.n_qubits)
    )
    return local + pair


def _encoding_unitary(values: np.ndarray, config: TFIMTransitionConfig) -> np.ndarray:
    operators = [
        _ry(config.encoding_angle_scale * float(value))
        for value in values
    ]
    return _kron_all(operators)


def tfim_transition_features(
    windows: np.ndarray,
    config: TFIMTransitionConfig | None = None,
) -> np.ndarray:
    """Return eight final-state observables for each ordered episode path.

    Each episode starts from |+>^4. At each time step, the four scaled channel
    values are encoded by local Ry rotations followed by one fixed TFIM evolution
    step. The state is carried forward through the full path.
    """
    cfg = config or TFIMTransitionConfig()
    scaled = scale_transition_windows(windows)
    evolution = make_tfim_evolution(cfg)
    observables = make_observables(cfg)

    plus = np.asarray([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    initial = _kron_all([plus] * cfg.n_qubits)
    features = np.empty((len(scaled), len(observables)), dtype=float)

    for episode_idx, window in enumerate(scaled):
        state = initial.copy()
        for values in window:
            state = evolution @ (_encoding_unitary(values, cfg) @ state)
        for observable_idx, observable in enumerate(observables):
            expectation = np.vdot(state, observable @ state)
            features[episode_idx, observable_idx] = float(np.real_if_close(expectation))

    return features
