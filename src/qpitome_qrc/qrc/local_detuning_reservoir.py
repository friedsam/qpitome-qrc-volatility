"""Static local-detuning Rydberg feature map.

This module complements ``rydberg_reservoir`` with a state-conditioned local
field. Each sample supplies one spatial detuning pattern over the atoms while
the global Rabi drive and global detuning remain fixed. The state is freshly
initialized for every sample; this is a static nonlinear feature map, not a
temporal reservoir.

Hamiltonian (angular-frequency units):

    H / hbar = (Omega / 2) sum_i sigma_i^x
               - Delta_g sum_i n_i
               - Delta_l sum_i h_i(x) n_i
               + sum_{i<j} V_ij n_i n_j

The local pattern ``h_i(x)`` is constrained to [0, 1]. On hardware, changing
``h_i`` changes the program; samples are not free shot repetitions of one
waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    _apply_global_rx_batch,
    _measure_features,
    precompute,
)


@dataclass(frozen=True)
class LocalDetuningConfig:
    """Configuration for one static local-detuning evolution."""

    reservoir: RydbergQRCConfig = RydbergQRCConfig(
        geometry="dual_chain",
        n_atoms_slow=4,
        n_atoms_fast=4,
        spacing_slow_um=9.0,
        spacing_fast_um=15.0,
        row_gap_um=14.0,
        observable_mode="n_nn",
        collect_anchor_features=False,
        memory_mode="memoryless",
        shots=None,
    )
    evolution_time_us: float = 0.55
    global_omega_rad_us: float = 6.0
    global_delta_rad_us: float = 6.0
    local_delta_rad_us: float = 4.0


def validate_local_patterns(patterns: np.ndarray, n_atoms: int) -> np.ndarray:
    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim != 2 or patterns.shape[1] != n_atoms:
        raise ValueError(
            f"Expected local patterns (samples, {n_atoms}), got {patterns.shape}"
        )
    if not np.all(np.isfinite(patterns)):
        raise ValueError("Local patterns contain non-finite values")
    if np.any(patterns < 0.0) or np.any(patterns > 1.0):
        raise ValueError("Local pattern coefficients must lie in [0, 1]")
    return patterns


def evolve_local_detuning_batch(
    local_patterns: np.ndarray,
    config: LocalDetuningConfig,
) -> np.ndarray:
    """Evolve one fresh state per local-detuning pattern and return features."""
    pre = precompute(config.reservoir)
    patterns = validate_local_patterns(local_patterns, pre.n_atoms)
    n_samples = len(patterns)
    dim = 2**pre.n_atoms

    states = np.zeros((n_samples, dim), dtype=complex)
    states[:, 0] = 1.0

    # Site-resolved occupation-weighted local pattern for every basis state.
    local_occupation = patterns @ pre.occ_bits.T
    diag = (
        pre.e_int[None, :]
        - config.global_delta_rad_us * pre.n_occ[None, :]
        - config.local_delta_rad_us * local_occupation
    )

    scale = max(
        float(pre.e_int.max(initial=0.0)),
        abs(config.global_delta_rad_us) + abs(config.local_delta_rad_us),
        abs(config.global_omega_rad_us),
        1e-9,
    )
    n_steps = int(
        np.ceil(
            scale
            * config.evolution_time_us
            / config.reservoir.max_phase_per_substep
        )
    )
    n_steps = int(
        np.clip(n_steps, 1, config.reservoir.max_substeps_per_segment)
    )
    dt = config.evolution_time_us / n_steps
    half = np.exp(-0.5j * dt * diag)
    full = half * half
    theta = np.full(n_samples, config.global_omega_rad_us * dt)

    states = states * half
    for step in range(n_steps):
        states = _apply_global_rx_batch(states, theta, pre.n_atoms)
        states = states * (half if step == n_steps - 1 else full)

    rng = (
        np.random.default_rng(config.reservoir.shot_seed)
        if config.reservoir.shots is not None
        else None
    )
    return _measure_features(states, pre, config.reservoir, rng)


def build_local_detuning_feature_matrix(
    local_patterns: np.ndarray,
    config: LocalDetuningConfig | None = None,
) -> np.ndarray:
    """Public feature-builder API for static local-detuning patterns."""
    return evolve_local_detuning_batch(
        local_patterns,
        config or LocalDetuningConfig(),
    )
