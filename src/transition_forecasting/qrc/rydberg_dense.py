from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class RydbergDenseConfig:
    n_atoms: int = 6
    omega: float = 1.0
    base_detuning: float = 0.0
    input_scale: float = 0.35
    local_scale: float = 0.25
    nearest_interaction: float = 1.0
    step_duration: float = 0.20


@lru_cache(maxsize=None)
def _single_site_operators(n_atoms: int) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    ident = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    n = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    xs: list[np.ndarray] = []
    ns: list[np.ndarray] = []
    for site in range(n_atoms):
        x_ops = [ident] * n_atoms
        n_ops = [ident] * n_atoms
        x_ops[site] = x
        n_ops[site] = n
        x_full = x_ops[0]
        n_full = n_ops[0]
        for op in x_ops[1:]:
            x_full = np.kron(x_full, op)
        for op in n_ops[1:]:
            n_full = np.kron(n_full, op)
        xs.append(x_full)
        ns.append(n_full)
    return tuple(xs), tuple(ns)


def initial_ground_state(n_atoms: int) -> np.ndarray:
    state = np.zeros(2**n_atoms, dtype=complex)
    state[0] = 1.0
    return state


def interaction_matrix(n_atoms: int, nearest_interaction: float) -> np.ndarray:
    matrix = np.zeros((n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = float(j - i)
            matrix[i, j] = nearest_interaction / distance**6
            matrix[j, i] = matrix[i, j]
    return matrix


def hamiltonian(
    global_input: float,
    local_pattern: np.ndarray,
    config: RydbergDenseConfig,
    *,
    interactions: bool = True,
) -> np.ndarray:
    if local_pattern.shape != (config.n_atoms,):
        raise ValueError(f"local_pattern must have shape {(config.n_atoms,)}")
    xs, ns = _single_site_operators(config.n_atoms)
    dim = 2**config.n_atoms
    result = np.zeros((dim, dim), dtype=complex)
    for site in range(config.n_atoms):
        result += 0.5 * config.omega * xs[site]
        detuning = (
            config.base_detuning
            + config.input_scale * float(global_input)
            + config.local_scale * float(local_pattern[site])
        )
        result -= detuning * ns[site]
    if interactions:
        weights = interaction_matrix(config.n_atoms, config.nearest_interaction)
        for i in range(config.n_atoms):
            for j in range(i + 1, config.n_atoms):
                result += weights[i, j] * (ns[i] @ ns[j])
    return result


def observables(state: np.ndarray, n_atoms: int) -> np.ndarray:
    _, ns = _single_site_operators(n_atoms)
    occupations = np.array([np.real(np.vdot(state, op @ state)) for op in ns], dtype=float)
    pair_values: list[float] = []
    connected: list[float] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pair = float(np.real(np.vdot(state, (ns[i] @ ns[j]) @ state)))
            pair_values.append(pair)
            connected.append(pair - occupations[i] * occupations[j])
    return np.concatenate([occupations, np.asarray(pair_values), np.asarray(connected)])


def evolve_sequence(
    sequence: np.ndarray,
    local_pattern: np.ndarray,
    config: RydbergDenseConfig,
    *,
    probe_steps: tuple[int, ...],
    interactions: bool = True,
    reset_each_step: bool = False,
) -> np.ndarray:
    values = np.asarray(sequence, dtype=float).reshape(-1)
    if not len(values):
        raise ValueError("sequence is empty")
    probes = tuple(sorted(set(int(step) for step in probe_steps)))
    if probes[-1] > len(values) or probes[0] < 1:
        raise ValueError("probe_steps must be within the sequence")
    state = initial_ground_state(config.n_atoms)
    features: list[np.ndarray] = []
    for step, value in enumerate(values, start=1):
        if reset_each_step:
            state = initial_ground_state(config.n_atoms)
        h = hamiltonian(value, local_pattern, config, interactions=interactions)
        state = expm(-1j * config.step_duration * h) @ state
        state /= np.linalg.norm(state)
        if step in probes:
            features.append(observables(state, config.n_atoms))
    return np.concatenate(features)
