#!/usr/bin/env python
"""
Tiny QRC prototype.

Goal:
- build a fixed quantum reservoir
- encode classical inputs as rotation angles
- extract quantum expectation-value features
- train only a classical readout

This is intentionally small and deterministic.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_reservoir_circuit(
    x: np.ndarray,
    reservoir_weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> QuantumCircuit:
    """Encode one input vector and apply a fixed random reservoir circuit."""
    qc = QuantumCircuit(n_qubits)

    # Input encoding: map each feature to RX/RZ angles.
    for q in range(n_qubits):
        angle = float(x[q])
        qc.rx(angle, q)
        qc.rz(0.5 * angle, q)

    # Fixed reservoir dynamics.
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(reservoir_weights[layer, q, 0]), q)
            qc.rz(float(reservoir_weights[layer, q, 1]), q)

        # Ring entanglement.
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)

    return qc


def z_observables(n_qubits: int) -> list[SparsePauliOp]:
    """Single-qubit Z observables: <Z0>, <Z1>, ..."""
    ops = []
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        # Qiskit Pauli strings are little-endian relative to qubit indexing.
        label[n_qubits - 1 - q] = "Z"
        ops.append(SparsePauliOp("".join(label)))
    return ops


def extract_qrc_features(
    X: np.ndarray,
    reservoir_weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> np.ndarray:
    """Return quantum reservoir feature matrix with shape (n_samples, n_qubits)."""
    obs = z_observables(n_qubits)
    features = np.zeros((X.shape[0], len(obs)), dtype=float)

    for i, x in enumerate(X):
        qc = build_reservoir_circuit(x, reservoir_weights, n_qubits, n_layers)
        psi = Statevector.from_instruction(qc)

        for j, op in enumerate(obs):
            features[i, j] = float(np.real(psi.expectation_value(op)))

    return features


def main() -> None:
    rng = np.random.default_rng(7)

    n_samples = 300
    n_qubits = 4
    n_layers = 2

    # Temporary synthetic data until the committed baseline data loader is ready.
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_qubits,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=0.8,
        random_state=7,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=7,
        stratify=y,
    )

    # Scale to bounded rotation-angle range.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.clip(X_train, -3.0, 3.0)
    X_test = np.clip(X_test, -3.0, 3.0)

    reservoir_weights = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=(n_layers, n_qubits, 2),
    )

    Phi_train = extract_qrc_features(
        X_train,
        reservoir_weights=reservoir_weights,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    Phi_test = extract_qrc_features(
        X_test,
        reservoir_weights=reservoir_weights,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )

    clf = LogisticRegression(class_weight="balanced", random_state=7)
    clf.fit(Phi_train, y_train)
    y_pred = clf.predict(Phi_test)

    print("Tiny QRC prototype")
    print("==================")
    print(f"n_samples: {n_samples}")
    print(f"n_qubits:  {n_qubits}")
    print(f"n_layers:  {n_layers}")
    print(f"Phi_train shape: {Phi_train.shape}")
    print(f"Phi_test shape:  {Phi_test.shape}")
    print()
    print(f"accuracy:          {accuracy_score(y_test, y_pred):.3f}")
    print(f"balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print()
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
    