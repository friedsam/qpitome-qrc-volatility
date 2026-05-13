#!/usr/bin/env python
"""
Tiny real-data QRC prototype.

Uses the same processed dataset, features, target, sequence construction,
chronological split, and evaluation metrics as scripts/run_esn_baseline.py.

Pipeline:
20 x 12 market-stress sequence
-> flatten to 240 values
-> StandardScaler + PCA(4)
-> 4-qubit QRC circuit
-> Z expectation features
-> LogisticRegression readout
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Allow importing from scripts/run_esn_baseline.py when executed as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_esn_baseline import (  # noqa: E402
    DATA_PATH,
    FEATURES,
    TARGET,
    chronological_split,
    evaluate,
    make_sequences,
)


def build_reservoir_circuit(
    x: np.ndarray,
    reservoir_weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> QuantumCircuit:
    """Build QRC circuit for one compressed input vector."""
    qc = QuantumCircuit(n_qubits)

    # Data encoding: one PCA component per qubit.
    for q in range(n_qubits):
        angle = float(x[q])
        qc.rx(angle, q)
        qc.rz(0.5 * angle, q)

    # Fixed random reservoir.
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(reservoir_weights[layer, q, 0]), q)
            qc.rz(float(reservoir_weights[layer, q, 1]), q)

        # Ring entanglement.
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)

    return qc


"""
def z_observables(n_qubits: int) -> list[SparsePauliOp]:
    #Single-qubit Z observables: <Z0>, ..., <Z(n-1)>.
    ops = []

    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "Z"
        ops.append(SparsePauliOp("".join(label)))

    return ops
"""

def z_observables(n_qubits: int) -> list[SparsePauliOp]:
    """Single-qubit Z and nearest-neighbor ZZ observables."""
    ops = []

    # <Zi>
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "Z"
        ops.append(SparsePauliOp("".join(label)))

    # <Zi Z(i+1)> on ring
    for q in range(n_qubits):
        r = (q + 1) % n_qubits
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "Z"
        label[n_qubits - 1 - r] = "Z"
        ops.append(SparsePauliOp("".join(label)))

    return ops

def extract_qrc_features(
    X_angles: np.ndarray,
    reservoir_weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> np.ndarray:
    """Extract QRC statevector expectation features."""
    obs = z_observables(n_qubits)
    Phi = np.zeros((X_angles.shape[0], len(obs)), dtype=float)

    for i, x in enumerate(X_angles):
        qc = build_reservoir_circuit(
            x=x,
            reservoir_weights=reservoir_weights,
            n_qubits=n_qubits,
            n_layers=n_layers,
        )
        psi = Statevector.from_instruction(qc)

        for j, op in enumerate(obs):
            Phi[i, j] = float(np.real(psi.expectation_value(op)))

    return Phi


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten sequence tensor: (n_samples, seq_len, input_dim) -> (n_samples, seq_len*input_dim)."""
    return X.reshape(X.shape[0], -1)


def main() -> None:
    rng = np.random.default_rng(42)

    n_qubits = 4
    n_layers = 2
    seq_len = 20

    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    X, y, dates = make_sequences(df, FEATURES, TARGET, seq_len=seq_len)

    (
        X_train,
        y_train,
        dates_train,
        X_val,
        y_val,
        dates_val,
        X_test,
        y_test,
        dates_test,
    ) = chronological_split(X, y, dates)

    print("Sequence shapes:")
    print("train:", X_train.shape, y_train.shape, dates_train.min(), dates_train.max())
    print("val:  ", X_val.shape, y_val.shape, dates_val.min(), dates_val.max())
    print("test: ", X_test.shape, y_test.shape, dates_test.min(), dates_test.max())

    print("\nLabel rates:")
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(name, dict(zip(*np.unique(yy, return_counts=True))))

    X_train_flat = flatten_sequences(X_train)
    X_val_flat = flatten_sequences(X_val)
    X_test_flat = flatten_sequences(X_test)

    # Compression: fit only on train to avoid leakage.
    compressor = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_qubits, random_state=42),
    )

    X_train_angles = compressor.fit_transform(X_train_flat)
    X_val_angles = compressor.transform(X_val_flat)
    X_test_angles = compressor.transform(X_test_flat)

    explained = compressor.named_steps["pca"].explained_variance_ratio_
    print("\nPCA compression:")
    print("components:", n_qubits)
    print("explained_variance_ratio:", np.round(explained, 4))
    print("cumulative_explained_variance:", float(np.sum(explained)))

    # Bound PCA values before using them as rotation angles.
    X_train_angles = np.clip(X_train_angles, -3.0, 3.0)
    X_val_angles = np.clip(X_val_angles, -3.0, 3.0)
    X_test_angles = np.clip(X_test_angles, -3.0, 3.0)

    reservoir_weights = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=(n_layers, n_qubits, 2),
    )

    print("\nExtracting QRC features...")
    Phi_train = extract_qrc_features(
        X_train_angles,
        reservoir_weights=reservoir_weights,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    Phi_val = extract_qrc_features(
        X_val_angles,
        reservoir_weights=reservoir_weights,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    Phi_test = extract_qrc_features(
        X_test_angles,
        reservoir_weights=reservoir_weights,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )

    print("Phi_train:", Phi_train.shape)
    print("Phi_val:  ", Phi_val.shape)
    print("Phi_test: ", Phi_test.shape)

    # Baseline 1: majority class.
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train_flat, y_train)

    for name, X_eval, yy in [
        ("Majority Validation", X_val_flat, y_val),
        ("Majority Test", X_test_flat, y_test),
    ]:
        y_pred = majority.predict(X_eval)
        y_score = np.full_like(yy, fill_value=float(np.mean(y_train)), dtype=float)
        evaluate(name, yy, y_pred, y_score)

    # Baseline 2: raw flattened PCA-compressed logistic regression.
    raw_readout = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    raw_readout.fit(X_train_angles, y_train)

    for name, X_eval, yy in [
        ("PCA-raw Validation", X_val_angles, y_val),
        ("PCA-raw Test", X_test_angles, y_test),
    ]:
        y_score = raw_readout.predict_proba(X_eval)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        evaluate(name, yy, y_pred, y_score)

    """
    # QRC readout.
    qrc_readout = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    qrc_readout.fit(Phi_train, y_train)

    for name, Phi, yy in [
        ("QRC Validation", Phi_val, y_val),
        ("QRC Test", Phi_test, y_test),
    ]:
        y_score = qrc_readout.predict_proba(Phi)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        evaluate(name, yy, y_pred, y_score)
    """

        # QRC readout.
    qrc_readout = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    qrc_readout.fit(Phi_train, y_train)

    for name, Phi, yy in [
        ("QRC Validation", Phi_val, y_val),
        ("QRC Test", Phi_test, y_test),
    ]:
        y_score = qrc_readout.predict_proba(Phi)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        evaluate(name, yy, y_pred, y_score)

    # Hybrid readout: preserve compressed PCA signal and append QRC observables.
    Phi_train_hybrid = np.hstack([X_train_angles, Phi_train])
    Phi_val_hybrid = np.hstack([X_val_angles, Phi_val])
    Phi_test_hybrid = np.hstack([X_test_angles, Phi_test])

    hybrid_readout = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    hybrid_readout.fit(Phi_train_hybrid, y_train)

    
    for name, Phi, yy in [
        ("PCA+QRC Validation", Phi_val_hybrid, y_val),
        ("PCA+QRC Test", Phi_test_hybrid, y_test),
    ]:
        y_score = hybrid_readout.predict_proba(Phi)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        evaluate(name, yy, y_pred, y_score)
    


    print("\nCurrent takeaway")
    print("================")
    print("Standalone QRC is weak.")
    print("PCA+QRC at threshold 0.5 slightly improves PR-AUC and reduces false positives vs PCA-only.")

if __name__ == "__main__":
    main()