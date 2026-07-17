"""Exact-state temporal Rydberg reservoir features for destination forecasting."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _kron_all(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def build_operators(n_atoms: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    n = np.array([[0.0, 0.0], [0.0, 1.0]])
    identity = np.eye(2)
    x_ops = [_kron_all([x if index == site else identity for index in range(n_atoms)]) for site in range(n_atoms)]
    n_ops = [_kron_all([n if index == site else identity for index in range(n_atoms)]) for site in range(n_atoms)]
    drive = sum(x_ops)
    number = sum(n_ops)
    pair_ops = [n_ops[index] @ n_ops[index + 1] for index in range(n_atoms - 1)]
    interaction = sum(pair_ops)
    return drive, number, n_ops, pair_ops, interaction


def precompute_unitaries(values: np.ndarray, drive: np.ndarray, number: np.ndarray, interaction: np.ndarray, *, omega: float, detuning_base: float, detuning_scale: float, interaction_strength: float, segment_time: float, quantization_levels: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    grid = np.linspace(-1.0, 1.0, quantization_levels)
    unique_indices = np.unique(np.argmin(np.abs(values[..., None] - grid), axis=-1))
    unitaries = {}
    for index in unique_indices:
        detuning = detuning_base + detuning_scale * float(grid[index])
        hamiltonian = 0.5 * omega * drive - detuning * number + interaction_strength * interaction
        unitaries[int(index)] = expm(-1j * segment_time * hamiltonian)
    return unitaries, grid


def reservoir_features(encoded: np.ndarray, n_ops: list[np.ndarray], pair_ops: list[np.ndarray], unitaries: dict[int, np.ndarray], grid: np.ndarray) -> np.ndarray:
    rows = []
    dimension = n_ops[0].shape[0]
    for episode in encoded:
        state = np.zeros(dimension, dtype=complex)
        state[0] = 1.0
        snapshots = []
        for week in episode:
            for value in week:
                index = int(np.argmin(np.abs(grid - value)))
                state = unitaries[index] @ state
                state /= np.linalg.norm(state)
            observables = [float(np.real(np.vdot(state, operator @ state))) for operator in [*n_ops, *pair_ops]]
            snapshots.append(observables)
        snapshots = np.asarray(snapshots)
        rows.append(np.r_[snapshots[-1], snapshots.mean(axis=0)])
    return np.asarray(rows)


def compute_feature_sets(encodings: dict[str, np.ndarray], *, n_atoms: int, omega: float, detuning_base: float, detuning_scale: float, interaction_strength: float, segment_time: float, quantization_levels: int) -> dict[str, np.ndarray]:
    drive, number, n_ops, pair_ops, interaction = build_operators(n_atoms)
    outputs = {}
    for name, encoded in encodings.items():
        unitaries, grid = precompute_unitaries(encoded, drive, number, interaction, omega=omega, detuning_base=detuning_base, detuning_scale=detuning_scale, interaction_strength=interaction_strength, segment_time=segment_time, quantization_levels=quantization_levels)
        outputs[name] = reservoir_features(encoded, n_ops, pair_ops, unitaries, grid)
    return outputs
