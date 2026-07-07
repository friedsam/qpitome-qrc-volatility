"""Frozen RF-QRC references copied from original Phase 3 experiment scripts.

Do not refactor this file. It exists only as a numerical oracle for regression
and migration tests while the production implementation is cleaned up.
"""

from __future__ import annotations

import math

import numpy as np


def ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex)


def apply_one(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    tensor = state.reshape([2] * n)
    tensor = np.moveaxis(tensor, q, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, q)
    return tensor.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    out = np.empty_like(state)
    for idx, amp in enumerate(state):
        dest = idx ^ (1 << target) if ((idx >> control) & 1) else idx
        out[dest] = amp
    return out


def apply_zz_phase(state: np.ndarray, i: int, j: int, theta: float, n: int) -> np.ndarray:
    out = state.copy()
    for idx in range(len(out)):
        zi = 1.0 if ((idx >> i) & 1) == 0 else -1.0
        zj = 1.0 if ((idx >> j) & 1) == 0 else -1.0
        out[idx] *= np.exp(-1j * theta * zi * zj)
    return out


def z_zz_features(state: np.ndarray, n: int) -> np.ndarray:
    probs = np.abs(state) ** 2
    zvals = np.empty((len(state), n), dtype=float)
    for idx in range(len(state)):
        for q in range(n):
            zvals[idx, q] = 1.0 if ((idx >> q) & 1) == 0 else -1.0
    z = probs @ zvals
    zz = [probs @ (zvals[:, i] * zvals[:, j]) for i in range(n) for j in range(i + 1, n)]
    return np.concatenate([z, np.asarray(zz)])


class RFQRCMap:
    def __init__(self, n_qubits: int, second_encoding: bool, entangler: str, input_scale: float, seed: int):
        self.n = n_qubits
        self.second_encoding = second_encoding
        self.entangler = entangler
        self.input_scale = input_scale
        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, 0.35, size=n_qubits)
        self.ry_angles = rng.normal(0.0, 0.35, size=n_qubits)
        self.zz_angles = np.triu(rng.normal(0.0, 0.35, size=(n_qubits, n_qubits)), 1)

    def _encode(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        for q, val in enumerate(u):
            state = apply_one(state, ry(float(self.input_scale * val)), q, self.n)
        return state

    def _entangle(self, state: np.ndarray) -> np.ndarray:
        if self.entangler == "all_pairs":
            pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        elif self.entangler == "ring":
            pairs = [(i, (i + 1) % self.n) for i in range(self.n)]
        else:
            pairs = []
        for i, j in pairs:
            state = apply_cnot(state, i, j, self.n)
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
        state = self._encode(state, u)
        state = self._entangle(state)
        if self.second_encoding:
            state = self._encode(state, u)
        state = self._random_layer(state)
        return z_zz_features(state, self.n)

    def transform(self, U: np.ndarray) -> np.ndarray:
        return np.vstack([self.one(u) for u in U])


class TimeMultiplexedRFQRCMap:
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

    def _random_layer(self, state: np.ndarray, scale: float = 1.0) -> np.ndarray:
        for q in range(self.n):
            state = apply_one(state, rz(float(scale * self.rz_angles[q])), q, self.n)
            state = apply_one(state, ry(float(scale * self.ry_angles[q])), q, self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                state = apply_zz_phase(state, i, j, float(scale * self.zz_angles[i, j]), self.n)
        return state

    def one(self, u: np.ndarray, max_virtual_nodes: int, layer_scale: float) -> np.ndarray:
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        state = self._encode(state, u, factor=1.0)
        state = self._ring_entangle(state)
        state = self._encode(state, u, factor=1.0)
        feats = []
        for _ in range(max_virtual_nodes):
            state = self._random_layer(state, scale=layer_scale)
            feats.append(z_zz_features(state, self.n))
        return np.concatenate(feats)

    def transform(self, U: np.ndarray, max_virtual_nodes: int, layer_scale: float) -> np.ndarray:
        return np.vstack([self.one(u, max_virtual_nodes, layer_scale) for u in U])
