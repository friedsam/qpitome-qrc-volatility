"""Fixed four-qubit Rydberg feature map used by the legacy Day-5 baseline."""

from __future__ import annotations

import numpy as np

N_QUBITS = 4
N_STATE = 2**N_QUBITS
OMEGA = 1.0
EVOLVE_TIME = 1.35
POSITIONS = np.array([0.0, 1.0, 2.15, 3.6])
DETUNING_SCALE = 0.65
INTERACTION_SCALE = 0.85
INPUT_CLIP = 3.0


def bit_value(index: int, qubit: int) -> int:
    """Return the computational-basis occupation of one qubit."""

    return 1 if (index & (1 << qubit)) else 0


def build_operators() -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int, np.ndarray]]]:
    """Construct Pauli-X, number, and pair-number operators."""

    x_ops: list[np.ndarray] = []
    n_ops: list[np.ndarray] = []
    nn_ops: list[tuple[int, int, np.ndarray]] = []
    for qubit in range(N_QUBITS):
        x = np.zeros((N_STATE, N_STATE), dtype=complex)
        n = np.zeros((N_STATE, N_STATE), dtype=complex)
        for index in range(N_STATE):
            x[index ^ (1 << qubit), index] = 1.0
            n[index, index] = bit_value(index, qubit)
        x_ops.append(x)
        n_ops.append(n)
    for i in range(N_QUBITS):
        for j in range(i + 1, N_QUBITS):
            nn_ops.append((i, j, n_ops[i] @ n_ops[j]))
    return x_ops, n_ops, nn_ops


X_OPS, N_OPS, NN_OPS = build_operators()
PAIR_V = {
    (i, j): INTERACTION_SCALE / abs(POSITIONS[i] - POSITIONS[j]) ** 6
    for i in range(N_QUBITS)
    for j in range(i + 1, N_QUBITS)
}


def rydberg_features(x: np.ndarray) -> np.ndarray:
    """Return occupations, pair occupations, entropy, and excitation count."""

    x = np.clip(np.asarray(x, dtype=float), -INPUT_CLIP, INPUT_CLIP)
    deltas = DETUNING_SCALE * x
    hamiltonian = np.zeros((N_STATE, N_STATE), dtype=complex)
    for qubit in range(N_QUBITS):
        hamiltonian += 0.5 * OMEGA * X_OPS[qubit]
        hamiltonian += -deltas[qubit] * N_OPS[qubit]
    for i, j, nn in NN_OPS:
        hamiltonian += PAIR_V[(i, j)] * nn

    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    psi0 = np.zeros(N_STATE, dtype=complex)
    psi0[0] = 1.0
    coeff = eigvecs.conj().T @ psi0
    psi = eigvecs @ (np.exp(-1j * eigvals * EVOLVE_TIME) * coeff)

    features: list[float] = []
    for qubit in range(N_QUBITS):
        features.append(float(np.real(np.vdot(psi, N_OPS[qubit] @ psi))))
    for _, _, nn in NN_OPS:
        features.append(float(np.real(np.vdot(psi, nn @ psi))))

    probs = np.abs(psi) ** 2
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    excitation_count = sum(features[:N_QUBITS])
    features.extend([entropy, excitation_count])
    return np.asarray(features, dtype=float)
