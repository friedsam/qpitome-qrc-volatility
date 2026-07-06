"""Frozen structured RF-QRC logic copied from the original Phase 3 runner.

This file intentionally remains independent of the production structured model.
Low-level statevector primitives were already frozen and tested separately.
"""

from __future__ import annotations

import numpy as np

from tests.legacy_rf_qrc_reference import (
    apply_cnot,
    apply_one,
    apply_zz_phase,
    ry,
    rz,
    z_zz_features,
)


def entangler_pairs(mode: str, n: int) -> list[tuple[int, int]]:
    k = n // 2
    level = list(range(k))
    rate = list(range(k, n))
    if mode == "ring":
        return [(i, (i + 1) % n) for i in range(n)]
    if mode == "cross_matched":
        return [(level[i], rate[i]) for i in range(k)]
    if mode == "cross_all":
        return [(i, j) for i in level for j in rate]
    if mode == "block_plus_cross":
        within_level = [(level[i], level[i + 1]) for i in range(k - 1)]
        within_rate = [(rate[i], rate[i + 1]) for i in range(k - 1)]
        cross = [(i, j) for i in level for j in rate]
        return within_level + within_rate + cross
    if mode == "none":
        return []
    raise ValueError(f"Unknown entangler mode: {mode}")


class StructuredLevelRateRFQRCMap:
    def __init__(
        self,
        n_qubits: int,
        entangler: str,
        input_scale: float,
        level_scale: float,
        rate_scale: float,
        random_scale: float,
        cross_zz_boost: float,
        weak_within_boost: float,
        seed: int,
    ) -> None:
        if n_qubits % 2 != 0:
            raise ValueError("n_qubits must be even.")
        self.n = n_qubits
        self.k = n_qubits // 2
        self.entangler = entangler
        self.input_scale = input_scale
        self.level_scale = level_scale
        self.rate_scale = rate_scale
        self.cross_zz_boost = cross_zz_boost
        self.weak_within_boost = weak_within_boost
        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.ry_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.zz_angles = np.triu(rng.normal(0.0, random_scale, size=(n_qubits, n_qubits)), 1)

    def _encode(self, state: np.ndarray, u: np.ndarray, second: bool = False) -> np.ndarray:
        second_factor = 0.65 if second else 1.0
        for q, val in enumerate(u):
            channel_scale = self.level_scale if q < self.k else self.rate_scale
            theta = self.input_scale * channel_scale * second_factor * float(val)
            state = apply_one(state, ry(theta), q, self.n)
        return state

    def _entangle(self, state: np.ndarray) -> np.ndarray:
        for i, j in entangler_pairs(self.entangler, self.n):
            state = apply_cnot(state, i, j, self.n)
        return state

    def _random_layer(self, state: np.ndarray) -> np.ndarray:
        for q in range(self.n):
            state = apply_one(state, rz(float(self.rz_angles[q])), q, self.n)
            state = apply_one(state, ry(float(self.ry_angles[q])), q, self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                is_cross = (i < self.k <= j) or (j < self.k <= i)
                boost = self.cross_zz_boost if is_cross else self.weak_within_boost
                state = apply_zz_phase(state, i, j, float(boost * self.zz_angles[i, j]), self.n)
        return state

    def one(self, u: np.ndarray) -> np.ndarray:
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        state = self._encode(state, u, second=False)
        state = self._entangle(state)
        state = self._encode(state, u, second=True)
        state = self._random_layer(state)
        return z_zz_features(state, self.n)

    def transform(self, U: np.ndarray) -> np.ndarray:
        return np.vstack([self.one(u) for u in U])
