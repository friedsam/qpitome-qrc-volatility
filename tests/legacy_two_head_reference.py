"""Frozen one-QRC/two-head logic copied from the original Phase 3 runner."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

from tests.legacy_rf_qrc_reference import (
    apply_cnot,
    apply_one,
    apply_zz_phase,
    ry,
    rz,
    z_zz_features,
)

SEED = 42


class RFQRCRingMap:
    def __init__(self, n_qubits: int, input_scale: float, random_scale: float, seed: int):
        self.n = n_qubits
        self.input_scale = input_scale
        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.ry_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.zz_angles = np.triu(rng.normal(0.0, random_scale, size=(n_qubits, n_qubits)), 1)

    def _encode(self, state: np.ndarray, u: np.ndarray, factor: float = 1.0) -> np.ndarray:
        for q, val in enumerate(u):
            state = apply_one(state, ry(float(factor * self.input_scale * val)), q, self.n)
        return state

    def _ring_entangle(self, state: np.ndarray) -> np.ndarray:
        for i in range(self.n):
            state = apply_cnot(state, i, (i + 1) % self.n, self.n)
        return state

    def _random_layer(self, state: np.ndarray) -> np.ndarray:
        for q in range(self.n):
            state = apply_one(state, rz(float(self.rz_angles[q])), q, self.n)
            state = apply_one(state, ry(float(self.ry_angles[q])), q, self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                state = apply_zz_phase(state, i, j, float(self.zz_angles[i, j]), self.n)
        return state

    def one(self, u: np.ndarray) -> np.ndarray:
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        state = self._encode(state, u, factor=1.0)
        state = self._ring_entangle(state)
        state = self._encode(state, u, factor=1.0)
        state = self._random_layer(state)
        return z_zz_features(state, self.n)

    def transform(self, U: np.ndarray) -> np.ndarray:
        return np.vstack([self.one(u) for u in U])


def regression_head(F: np.ndarray, y: np.ndarray, split: np.ndarray, alpha: float) -> np.ndarray:
    train = split == "train"
    scaler = StandardScaler().fit(F[train])
    Fs = scaler.transform(F)
    model = Ridge(alpha=alpha).fit(Fs[train], np.log(np.maximum(y[train], 1e-8)))
    return np.maximum(np.exp(model.predict(Fs)), 1e-8)


def best_threshold_from_val(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    if len(thr) == 0:
        return 0.5, 0.0
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx]), float(f1[idx])


def classifier_head(
    F: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    threshold: float,
    C: float,
) -> tuple[np.ndarray, float]:
    train = split == "train"
    val = split == "val"
    event = y >= threshold
    scaler = StandardScaler().fit(F[train])
    Fs = scaler.transform(F)
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    clf.fit(Fs[train], event[train])
    prob = clf.predict_proba(Fs)[:, 1]
    decision_threshold, _ = best_threshold_from_val(event[val], prob[val])
    return prob, decision_threshold
