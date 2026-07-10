"""Static structured Rydberg interaction map for small-sample branch prediction.

This is intentionally not a temporal reservoir. It maps the six standardized
``state_plus_motion`` variables into a fixed six-atom Rydberg evolution and
returns six nearest-neighbor connected correlations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class RydbergInteractionConfig:
    n_atoms: int = 6
    omega: float = 1.0
    detuning: float = 0.8
    interaction_v: float = 1.5
    dt: float = 0.45
    depth: int = 3
    encoding_scale: float = 1.0


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


def _pair(
    left: np.ndarray,
    left_site: int,
    right: np.ndarray,
    right_site: int,
    n_atoms: int,
) -> np.ndarray:
    operators = [I2] * n_atoms
    operators[left_site] = left
    operators[right_site] = right
    return _kron_all(operators)


def make_rydberg_evolution(
    config: RydbergInteractionConfig | None = None,
) -> np.ndarray:
    """Build one fixed six-atom ring Rydberg evolution operator."""
    cfg = config or RydbergInteractionConfig()
    if cfg.n_atoms != 6:
        raise ValueError("Small-sample interaction map is fixed to six atoms")

    dim = 2 ** cfg.n_atoms
    hamiltonian = np.zeros((dim, dim), dtype=complex)

    for site in range(cfg.n_atoms):
        hamiltonian += 0.5 * cfg.omega * _single(X, site, cfg.n_atoms)
        hamiltonian -= cfg.detuning * _single(N, site, cfg.n_atoms)

        neighbor = (site + 1) % cfg.n_atoms
        hamiltonian += cfg.interaction_v * _pair(
            N,
            site,
            N,
            neighbor,
            cfg.n_atoms,
        )

    return expm(-1.0j * cfg.dt * hamiltonian)


def _encoding_phase(values: np.ndarray, config: RydbergInteractionConfig) -> np.ndarray:
    """Diagonal local number-phase encoding for one six-variable state vector."""
    dim = 2 ** config.n_atoms
    phases = np.ones(dim, dtype=complex)
    for basis_index in range(dim):
        phase = 0.0
        for site, value in enumerate(values):
            bit = (basis_index >> (config.n_atoms - 1 - site)) & 1
            if bit:
                phase += config.encoding_scale * float(value)
        phases[basis_index] = np.exp(-1.0j * phase)
    return phases


def _connected_ring_correlations(
    state: np.ndarray,
    config: RydbergInteractionConfig,
) -> np.ndarray:
    local = []
    for site in range(config.n_atoms):
        op = _single(N, site, config.n_atoms)
        local.append(float(np.real_if_close(np.vdot(state, op @ state))))

    correlations = np.empty(config.n_atoms, dtype=float)
    for site in range(config.n_atoms):
        neighbor = (site + 1) % config.n_atoms
        op = _pair(N, site, N, neighbor, config.n_atoms)
        pair_value = float(np.real_if_close(np.vdot(state, op @ state)))
        correlations[site] = pair_value - local[site] * local[neighbor]
    return correlations


def rydberg_interaction_features(
    standardized_inputs: np.ndarray,
    config: RydbergInteractionConfig | None = None,
) -> np.ndarray:
    """Map standardized six-variable states to six connected correlations.

    Inputs are expected to be standardized outside this function using training
    data only. Values are clipped to [-3, 3] before encoding.
    """
    cfg = config or RydbergInteractionConfig()
    values = np.asarray(standardized_inputs, dtype=float)
    if values.ndim != 2 or values.shape[1] != cfg.n_atoms:
        raise ValueError(
            f"Expected shape (sample, {cfg.n_atoms}); got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Rydberg interaction inputs contain non-finite values")

    clipped = np.clip(values, -3.0, 3.0)
    evolution = make_rydberg_evolution(cfg)
    initial = np.zeros(2 ** cfg.n_atoms, dtype=complex)
    initial[0] = 1.0

    features = np.empty((len(clipped), cfg.n_atoms), dtype=float)
    for row, sample in enumerate(clipped):
        state = initial.copy()
        phase = _encoding_phase(sample, cfg)
        for _ in range(cfg.depth):
            state = phase * state
            state = evolution @ state
        features[row] = _connected_ring_correlations(state, cfg)
    return features


def classical_ring_products(standardized_inputs: np.ndarray) -> np.ndarray:
    """Matched six-feature classical nonlinear interaction control."""
    values = np.asarray(standardized_inputs, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected shape (sample, 6); got {values.shape}")
    clipped = np.clip(values, -3.0, 3.0)
    return np.column_stack(
        [
            clipped[:, site] * clipped[:, (site + 1) % 6]
            for site in range(6)
        ]
    )
