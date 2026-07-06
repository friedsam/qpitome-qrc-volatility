"""Recurrence-free quantum reservoir feature maps used in Phase 3.

This module extracts the simulator logic that originally lived inside the
Phase 3 RF-QRC experiment scripts. The implementation intentionally preserves
legacy numerical behavior so historical results can be reproduced exactly
before any scientific or performance changes are considered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Entangler = Literal["none", "all_pairs", "ring"]


@dataclass(frozen=True)
class RFQRCConfig:
    """Configuration for the recurrence-free RF-QRC feature map."""

    n_qubits: int
    second_encoding: bool = False
    entangler: Entangler = "all_pairs"
    input_scale: float = math.pi / 3
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.entangler not in {"none", "all_pairs", "ring"}:
            raise ValueError(f"Unsupported entangler: {self.entangler!r}")


def ry(theta: float) -> np.ndarray:
    """Return the single-qubit Y-rotation used by the legacy RF-QRC code."""

    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    """Return the single-qubit Z-rotation used by the legacy RF-QRC code."""

    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def apply_one(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply a one-qubit gate using the legacy tensor-axis convention."""

    tensor = state.reshape([2] * n)
    tensor = np.moveaxis(tensor, q, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, q)
    return tensor.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    """Apply CNOT using the legacy bit-index convention."""

    out = np.empty_like(state)
    for idx, amp in enumerate(state):
        dest = idx ^ (1 << target) if ((idx >> control) & 1) else idx
        out[dest] = amp
    return out


def apply_zz_phase(
    state: np.ndarray,
    i: int,
    j: int,
    theta: float,
    n: int,
) -> np.ndarray:
    """Apply exp(-i theta Z_i Z_j) exactly as in the legacy scripts."""

    out = state.copy()
    for idx in range(len(out)):
        zi = 1.0 if ((idx >> i) & 1) == 0 else -1.0
        zj = 1.0 if ((idx >> j) & 1) == 0 else -1.0
        out[idx] *= np.exp(-1j * theta * zi * zj)
    return out


def z_zz_features(state: np.ndarray, n: int) -> np.ndarray:
    """Return all one-body Z and all-pairs ZZ expectation values."""

    probs = np.abs(state) ** 2
    zvals = np.empty((len(state), n), dtype=float)
    for idx in range(len(state)):
        for q in range(n):
            zvals[idx, q] = 1.0 if ((idx >> q) & 1) == 0 else -1.0
    z = probs @ zvals
    zz = [
        probs @ (zvals[:, i] * zvals[:, j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return np.concatenate([z, np.asarray(zz)])


def entangler_pairs(mode: Entangler, n_qubits: int) -> list[tuple[int, int]]:
    """Return the ordered CNOT pairs used by an RF-QRC entangler."""

    if mode == "all_pairs":
        return [
            (i, j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        ]
    if mode == "ring":
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    if mode == "none":
        return []
    raise ValueError(f"Unsupported entangler: {mode!r}")


class RFQRCMap:
    """Deterministic recurrence-free quantum feature map."""

    def __init__(
        self,
        n_qubits: int,
        second_encoding: bool,
        entangler: Entangler,
        input_scale: float,
        seed: int,
    ) -> None:
        self.config = RFQRCConfig(
            n_qubits=n_qubits,
            second_encoding=second_encoding,
            entangler=entangler,
            input_scale=input_scale,
            seed=seed,
        )
        self.n = n_qubits
        self.second_encoding = second_encoding
        self.entangler = entangler
        self.input_scale = input_scale

        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, 0.35, size=n_qubits)
        self.ry_angles = rng.normal(0.0, 0.35, size=n_qubits)
        self.zz_angles = np.triu(
            rng.normal(0.0, 0.35, size=(n_qubits, n_qubits)),
            1,
        )

    @classmethod
    def from_config(cls, config: RFQRCConfig) -> "RFQRCMap":
        return cls(
            config.n_qubits,
            config.second_encoding,
            config.entangler,
            config.input_scale,
            config.seed,
        )

    @property
    def n_features(self) -> int:
        """Number of Z plus all-pairs ZZ readout features."""

        return self.n + self.n * (self.n - 1) // 2

    def _encode(
        self,
        state: np.ndarray,
        u: np.ndarray,
        factor: float = 1.0,
    ) -> np.ndarray:
        if len(u) != self.n:
            raise ValueError(f"Expected {self.n} inputs, got {len(u)}")
        for q, val in enumerate(u):
            state = apply_one(
                state,
                ry(float(factor * self.input_scale * val)),
                q,
                self.n,
            )
        return state

    def _entangle(self, state: np.ndarray) -> np.ndarray:
        for i, j in entangler_pairs(self.entangler, self.n):
            state = apply_cnot(state, i, j, self.n)
        return state

    def _random_layer(self, state: np.ndarray, scale: float = 1.0) -> np.ndarray:
        for q in range(self.n):
            state = apply_one(
                state,
                rz(float(scale * self.rz_angles[q])),
                q,
                self.n,
            )
            state = apply_one(
                state,
                ry(float(scale * self.ry_angles[q])),
                q,
                self.n,
            )
        for i in range(self.n):
            for j in range(i + 1, self.n):
                state = apply_zz_phase(
                    state,
                    i,
                    j,
                    float(scale * self.zz_angles[i, j]),
                    self.n,
                )
        return state

    def state(self, u: np.ndarray) -> np.ndarray:
        """Return the final statevector for one input vector."""

        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        state = self._encode(state, np.asarray(u, dtype=float))
        state = self._entangle(state)
        if self.second_encoding:
            state = self._encode(state, np.asarray(u, dtype=float))
        return self._random_layer(state)

    def one(self, u: np.ndarray) -> np.ndarray:
        """Transform one input vector into Z/ZZ features."""

        return z_zz_features(self.state(u), self.n)

    def transform(self, U: np.ndarray) -> np.ndarray:
        """Transform a two-dimensional input matrix row by row."""

        values = np.asarray(U, dtype=float)
        if values.ndim != 2:
            raise ValueError("U must be a two-dimensional array")
        if values.shape[1] != self.n:
            raise ValueError(f"Expected {self.n} columns, got {values.shape[1]}")
        return np.vstack([self.one(u) for u in values])


class TimeMultiplexedRFQRCMap(RFQRCMap):
    """RF-QRC ring with repeated fixed random layers and virtual-node readout."""

    def __init__(
        self,
        n_qubits: int,
        input_scale: float,
        random_scale: float,
        seed: int,
    ) -> None:
        self.n = n_qubits
        self.second_encoding = True
        self.entangler = "ring"
        self.input_scale = input_scale
        self.config = RFQRCConfig(
            n_qubits=n_qubits,
            second_encoding=True,
            entangler="ring",
            input_scale=input_scale,
            seed=seed,
        )
        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.ry_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.zz_angles = np.triu(
            rng.normal(0.0, random_scale, size=(n_qubits, n_qubits)),
            1,
        )

    def one(
        self,
        u: np.ndarray,
        max_virtual_nodes: int,
        layer_scale: float,
    ) -> np.ndarray:
        """Return concatenated Z/ZZ features after each virtual node."""

        if max_virtual_nodes < 1:
            raise ValueError("max_virtual_nodes must be positive")
        values = np.asarray(u, dtype=float)
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        state = self._encode(state, values, factor=1.0)
        state = self._entangle(state)
        state = self._encode(state, values, factor=1.0)

        features = []
        for _ in range(max_virtual_nodes):
            state = self._random_layer(state, scale=layer_scale)
            features.append(z_zz_features(state, self.n))
        return np.concatenate(features)

    def transform(
        self,
        U: np.ndarray,
        max_virtual_nodes: int,
        layer_scale: float,
    ) -> np.ndarray:
        """Transform every input row into concatenated virtual-node features."""

        values = np.asarray(U, dtype=float)
        if values.ndim != 2:
            raise ValueError("U must be a two-dimensional array")
        if values.shape[1] != self.n:
            raise ValueError(f"Expected {self.n} columns, got {values.shape[1]}")
        return np.vstack(
            [
                self.one(u, max_virtual_nodes, layer_scale)
                for u in values
            ]
        )
