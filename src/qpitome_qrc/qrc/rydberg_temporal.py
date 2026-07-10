"""Fixed four-atom temporal Rydberg reservoir for matched relapse diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from qpitome_qrc.qrc.tfim_transition import scale_transition_windows


@dataclass(frozen=True)
class RydbergTemporalConfig:
    n_atoms: int = 4
    omega: float = 1.0
    detuning: float = 0.8
    interaction_v: float = 1.5
    dt: float = 0.35
    encoding_angle_scale: float = np.pi / 2.0


I2 = np.eye(2, dtype=complex)
X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
N = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=complex)


def _kron_all(operators: list[np.ndarray]) -> np.ndarray:
    out = operators[0]
    for operator in operators[1:]:
        out = np.kron(out, operator)
    return out


def _single(operator: np.ndarray, site: int, n_atoms: int) -> np.ndarray:
    operators = [I2] * n_atoms
    operators[site] = operator
    return _kron_all(operators)


def _pair(left_site: int, right_site: int, n_atoms: int) -> np.ndarray:
    operators = [I2] * n_atoms
    operators[left_site] = N
    operators[right_site] = N
    return _kron_all(operators)


def make_rydberg_temporal_evolution(
    config: RydbergTemporalConfig | None = None,
) -> np.ndarray:
    cfg = config or RydbergTemporalConfig()
    if cfg.n_atoms != 4:
        raise ValueError("Matched temporal Rydberg control is fixed to four atoms")
    dim = 2 ** cfg.n_atoms
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for site in range(cfg.n_atoms):
        hamiltonian += 0.5 * cfg.omega * _single(X, site, cfg.n_atoms)
        hamiltonian -= cfg.detuning * _single(N, site, cfg.n_atoms)
        neighbor = (site + 1) % cfg.n_atoms
        hamiltonian += cfg.interaction_v * _pair(site, neighbor, cfg.n_atoms)
    return expm(-1.0j * cfg.dt * hamiltonian)


def _encoding_phase(values: np.ndarray, config: RydbergTemporalConfig) -> np.ndarray:
    dim = 2 ** config.n_atoms
    phases = np.ones(dim, dtype=complex)
    for basis_index in range(dim):
        phase = 0.0
        for site, value in enumerate(values):
            bit = (basis_index >> (config.n_atoms - 1 - site)) & 1
            if bit:
                phase += config.encoding_angle_scale * float(value)
        phases[basis_index] = np.exp(-1.0j * phase)
    return phases


def rydberg_temporal_features(
    windows: np.ndarray,
    config: RydbergTemporalConfig | None = None,
) -> np.ndarray:
    """Return 4 occupations and 4 connected ring correlations after 40 steps."""
    cfg = config or RydbergTemporalConfig()
    scaled = scale_transition_windows(windows)
    evolution = make_rydberg_temporal_evolution(cfg)
    initial = np.zeros(2 ** cfg.n_atoms, dtype=complex)
    initial[0] = 1.0
    local_ops = [_single(N, site, cfg.n_atoms) for site in range(cfg.n_atoms)]
    pair_ops = [_pair(site, (site + 1) % cfg.n_atoms, cfg.n_atoms) for site in range(cfg.n_atoms)]

    features = np.empty((len(scaled), 8), dtype=float)
    for row, window in enumerate(scaled):
        state = initial.copy()
        for values in window:
            state = _encoding_phase(values, cfg) * state
            state = evolution @ state
        local = np.asarray(
            [float(np.real_if_close(np.vdot(state, op @ state))) for op in local_ops]
        )
        connected = np.empty(4, dtype=float)
        for site, op in enumerate(pair_ops):
            pair_value = float(np.real_if_close(np.vdot(state, op @ state)))
            connected[site] = pair_value - local[site] * local[(site + 1) % 4]
        features[row] = np.concatenate([local, connected])
    return features
