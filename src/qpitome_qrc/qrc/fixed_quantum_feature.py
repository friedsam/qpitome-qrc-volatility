"""Fixed four-qubit circuit feature map used by the legacy Day-5 baseline."""

from __future__ import annotations

import numpy as np

N_QUBITS = 4
N_STATE = 2**N_QUBITS
N_LAYERS = 2
INPUT_CLIP = 3.0
ENTANGLING_PAIRS = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3))


def apply_ry(state: np.ndarray, qubit: int, theta: float) -> np.ndarray:
    """Apply an RY rotation to one statevector qubit."""

    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    out = state.copy()
    bit = 1 << qubit
    for index in range(N_STATE):
        if index & bit:
            continue
        partner = index | bit
        a = state[index]
        b = state[partner]
        out[index] = cosine * a - sine * b
        out[partner] = sine * a + cosine * b
    return out


def apply_rx(state: np.ndarray, qubit: int, theta: float) -> np.ndarray:
    """Apply an RX rotation to one statevector qubit."""

    cosine = np.cos(theta / 2.0)
    sine = -1j * np.sin(theta / 2.0)
    out = state.copy()
    bit = 1 << qubit
    for index in range(N_STATE):
        if index & bit:
            continue
        partner = index | bit
        a = state[index]
        b = state[partner]
        out[index] = cosine * a + sine * b
        out[partner] = sine * a + cosine * b
    return out


def apply_zz_phase(
    state: np.ndarray,
    q1: int,
    q2: int,
    theta: float,
) -> np.ndarray:
    """Apply the fixed ZZ phase interaction to two qubits."""

    out = state.copy()
    bit1 = 1 << q1
    bit2 = 1 << q2
    for index in range(N_STATE):
        z1 = -1 if index & bit1 else 1
        z2 = -1 if index & bit2 else 1
        out[index] *= np.exp(-0.5j * theta * z1 * z2)
    return out


def expectation_z(state: np.ndarray, qubit: int) -> float:
    """Return the Pauli-Z expectation for one qubit."""

    bit = 1 << qubit
    probabilities = np.abs(state) ** 2
    signs = np.array([-1 if index & bit else 1 for index in range(N_STATE)])
    return float(np.sum(signs * probabilities).real)


def expectation_x(state: np.ndarray, qubit: int) -> float:
    """Return the Pauli-X expectation for one qubit."""

    bit = 1 << qubit
    total = 0.0j
    for index in range(N_STATE):
        total += np.conj(state[index]) * state[index ^ bit]
    return float(total.real)


def expectation_zz(state: np.ndarray, q1: int, q2: int) -> float:
    """Return the Pauli-ZZ expectation for a qubit pair."""

    bit1 = 1 << q1
    bit2 = 1 << q2
    probabilities = np.abs(state) ** 2
    signs = np.array([
        (-1 if index & bit1 else 1) * (-1 if index & bit2 else 1)
        for index in range(N_STATE)
    ])
    return float(np.sum(signs * probabilities).real)


def quantum_features(x: np.ndarray) -> np.ndarray:
    """Return fixed Z, X, and ZZ observables for one four-value input row."""

    x = np.clip(np.asarray(x, dtype=float), -INPUT_CLIP, INPUT_CLIP)
    state = np.zeros(N_STATE, dtype=complex)
    state[0] = 1.0
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            state = apply_ry(state, qubit, 0.85 * x[qubit] + 0.17 * layer)
        for q1, q2 in ENTANGLING_PAIRS:
            state = apply_zz_phase(state, q1, q2, 0.35 + 0.08 * (q1 + q2))
        for qubit in range(N_QUBITS):
            state = apply_rx(state, qubit, 0.23 + 0.05 * qubit)

    features: list[float] = []
    features.extend(expectation_z(state, qubit) for qubit in range(N_QUBITS))
    features.extend(expectation_x(state, qubit) for qubit in range(N_QUBITS))
    features.extend(
        expectation_zz(state, i, j)
        for i in range(N_QUBITS)
        for j in range(i + 1, N_QUBITS)
    )
    return np.asarray(features, dtype=float)
