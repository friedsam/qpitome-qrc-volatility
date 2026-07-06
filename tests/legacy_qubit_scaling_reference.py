"""Frozen RF-QRC logic from the Phase 3 qubit-scaling diagnostic."""

from __future__ import annotations

import numpy as np

from tests.legacy_rf_qrc_reference import (
    apply_cnot,
    apply_one,
    apply_zz_phase,
    ry,
    rz,
)


def selected_z_zz_features(state: np.ndarray, n: int, zz_mode: str) -> np.ndarray:
    probs = np.abs(state) ** 2
    zvals = np.empty((len(state), n), dtype=float)
    for idx in range(len(state)):
        for q in range(n):
            zvals[idx, q] = 1.0 if ((idx >> q) & 1) == 0 else -1.0
    z = probs @ zvals
    pairs: list[tuple[int, int]] = []
    if zz_mode in {"ring", "ring_plus_next"}:
        pairs.extend([(i, (i + 1) % n) for i in range(n)])
    if zz_mode == "ring_plus_next":
        pairs.extend([(i, (i + 2) % n) for i in range(n)])
    if zz_mode == "all":
        pairs.extend([(i, j) for i in range(n) for j in range(i + 1, n)])
    uniq = []
    seen = set()
    for i, j in pairs:
        a, b = sorted((i, j))
        if (a, b) not in seen:
            seen.add((a, b))
            uniq.append((a, b))
    zz = [probs @ (zvals[:, i] * zvals[:, j]) for i, j in uniq]
    return np.concatenate([z, np.asarray(zz, dtype=float)])


class RFQRCRingMap:
    def __init__(
        self,
        n_qubits: int,
        input_scale: float,
        random_scale: float,
        seed: int,
        zz_mode: str,
    ) -> None:
        self.n = n_qubits
        self.input_scale = input_scale
        self.zz_mode = zz_mode
        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.ry_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.zz_angles = np.triu(
            rng.normal(0.0, random_scale, size=(n_qubits, n_qubits)),
            1,
        )

    def one(self, u: np.ndarray) -> np.ndarray:
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        for q, val in enumerate(u):
            state = apply_one(state, ry(float(self.input_scale * val)), q, self.n)
        for i in range(self.n):
            state = apply_cnot(state, i, (i + 1) % self.n, self.n)
        for q, val in enumerate(u):
            state = apply_one(state, ry(float(self.input_scale * val)), q, self.n)
        for q in range(self.n):
            state = apply_one(state, rz(float(self.rz_angles[q])), q, self.n)
            state = apply_one(state, ry(float(self.ry_angles[q])), q, self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                state = apply_zz_phase(state, i, j, float(self.zz_angles[i, j]), self.n)
        return selected_z_zz_features(state, self.n, self.zz_mode)

    def transform(self, U: np.ndarray) -> np.ndarray:
        return np.vstack([self.one(u) for u in U])
