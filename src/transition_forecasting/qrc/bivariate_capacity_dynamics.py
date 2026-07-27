from __future__ import annotations

import itertools
from typing import Literal

import numpy as np

from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    probabilities_to_occupations,
    probabilities_to_symmetric_modes,
    validate_probe_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _fresh_states,
    _resolve_probe_steps,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)

FeatureBankName = Literal[
    "raw_input_linear",
    "six_mode_density_curvature",
    "nine_mode_symmetric",
    "all_occupations",
    "full_one_two_body",
]

QRC_FEATURE_BANKS: tuple[FeatureBankName, ...] = (
    "six_mode_density_curvature",
    "nine_mode_symmetric",
    "all_occupations",
    "full_one_two_body",
)


def _apply_global_rxy_batch(
    states: np.ndarray,
    angles: np.ndarray,
    phases: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Apply a sample-specific global rotation around a phase-selected XY axis."""

    state = np.asarray(states, dtype=complex)
    angle = np.asarray(angles, dtype=float)
    phase = np.asarray(phases, dtype=float)
    if state.ndim != 2 or state.shape[1] != 2**n_atoms:
        raise ValueError("states have an incompatible shape")
    if angle.shape != (len(state),) or phase.shape != (len(state),):
        raise ValueError("angles and phases must align with state rows")

    cosine = np.cos(angle / 2.0)
    sine = np.sin(angle / 2.0)
    phase_plus = np.exp(1j * phase)
    phase_minus = np.conjugate(phase_plus)
    result = state
    for qubit in range(n_atoms):
        left = 2**qubit
        right = 2 ** (n_atoms - 1 - qubit)
        view = result.reshape(-1, left, 2, right)
        state_zero = view[:, :, 0, :].copy()
        state_one = view[:, :, 1, :].copy()
        c = cosine[:, None, None]
        s = sine[:, None, None]
        plus = phase_plus[:, None, None]
        minus = phase_minus[:, None, None]
        view[:, :, 0, :] = c * state_zero - 1j * s * plus * state_one
        view[:, :, 1, :] = -1j * s * minus * state_zero + c * state_one
        result = view.reshape(-1, 2**n_atoms)
    return result


def _evolve_segment_batch(
    states: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
    phase: np.ndarray,
    duration_us: float,
    reservoir: TemporalRydbergChainConfig,
    precomputed: object,
    *,
    interactions: bool,
) -> np.ndarray:
    """Strang-split one arbitrary-duration global AHS segment."""

    if duration_us <= 0:
        raise ValueError("duration_us must be positive")
    omega_values = np.asarray(omega, dtype=float)
    delta_values = np.asarray(delta, dtype=float)
    phase_values = np.asarray(phase, dtype=float)
    samples = len(states)
    if any(array.shape != (samples,) for array in (omega_values, delta_values, phase_values)):
        raise ValueError("drive arrays must align with state rows")

    interaction_energy = (
        np.asarray(precomputed.interaction_energy, dtype=float)
        if interactions
        else np.zeros_like(precomputed.interaction_energy, dtype=float)
    )
    scale = max(
        float(np.max(np.abs(omega_values), initial=0.0)),
        float(np.max(np.abs(delta_values), initial=0.0)),
        float(np.max(np.abs(interaction_energy), initial=0.0)),
        1e-12,
    )
    substeps = int(
        np.clip(
            np.ceil(scale * float(duration_us) / reservoir.max_phase_per_substep),
            1,
            reservoir.max_substeps_per_step,
        )
    )
    dt = float(duration_us) / substeps
    diagonal = (
        interaction_energy[None, :]
        - delta_values[:, None] * precomputed.total_occupation[None, :]
    )
    half = np.exp(-0.5j * dt * diagonal)
    full = half * half
    angles = omega_values * dt

    result = np.asarray(states, dtype=complex) * half
    for substep in range(substeps):
        result = _apply_global_rxy_batch(
            result,
            angles,
            phase_values,
            int(precomputed.n_atoms),
        )
        result *= half if substep == substeps - 1 else full
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise RuntimeError("sequential segment normalization failed")
    return result / norm


def evolve_sequential_probe_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float,
    first_slot_fraction: float,
    second_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Encode u1 and u2 in consecutive noncommuting global pulse slots.

    Slot A uses u1 in global detuning with an X-axis drive. Slot B uses u2 in
    global Rabi amplitude with a phase-shifted drive. Slot durations sum to the
    incumbent per-observation duration.
    """

    reservoir.validate()
    geometry.validate()
    values = np.asarray(windows, dtype=float)
    if (
        values.ndim != 3
        or values.shape[2] != 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("sequential probability extraction requires six exact atoms")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")
    if not 0.1 <= first_slot_fraction <= 0.9:
        raise ValueError("first_slot_fraction must lie in [0.1, 0.9]")
    if not np.isfinite(second_phase_rad):
        raise ValueError("second_phase_rad must be finite")

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    samples, steps, _ = values.shape
    first_duration = reservoir.step_duration_us * float(first_slot_fraction)
    second_duration = reservoir.step_duration_us - first_duration
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    interactions = interaction_scale > 0
    states = _fresh_states(samples, precomputed.n_atoms)
    blocks: list[np.ndarray] = []

    for step in range(steps):
        u1 = values[:, step, 0]
        u2 = values[:, step, 1]
        states = _evolve_segment_batch(
            states,
            np.full(samples, reservoir.omega_base_rad_us, dtype=float),
            reservoir.delta_center_rad_us + reservoir.delta_span_rad_us * u1,
            np.zeros(samples, dtype=float),
            first_duration,
            reservoir,
            precomputed,
            interactions=interactions,
        )
        omega_second = reservoir.omega_base_rad_us * (
            1.0 + reservoir.omega_mod_fraction * u2
        )
        if np.any(omega_second <= 0):
            raise ValueError("sequential amplitude encoding became nonpositive")
        states = _evolve_segment_batch(
            states,
            omega_second,
            np.full(samples, reservoir.delta_center_rad_us, dtype=float),
            np.full(samples, float(second_phase_rad), dtype=float),
            second_duration,
            reservoir,
            precomputed,
            interactions=interactions,
        )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1),
        n_atoms=precomputed.n_atoms,
    )
    metadata = {
        "encoding": "sequential_noncommuting",
        "probe_steps": [int(value) for value in probe_steps],
        "interaction_scale": float(interaction_scale),
        "first_slot_duration_us": float(first_duration),
        "second_slot_duration_us": float(second_duration),
        "second_slot_phase_rad": float(second_phase_rad),
        "total_evolution_time_us": float(steps * reservoir.step_duration_us),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }
    return stacked, metadata


def probabilities_to_full_low_order(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return all six occupations, all 15 pairs, and all 15 connected pairs."""

    values = validate_probe_probabilities(probabilities)
    occupations = probabilities_to_occupations(values)
    pairs = tuple(itertools.combinations(range(6), 2))
    states = np.arange(64)
    bits = np.stack(
        [((states >> (5 - site)) & 1) for site in range(6)],
        axis=1,
    ).astype(float)
    pair_bits = np.stack(
        [bits[:, left] * bits[:, right] for left, right in pairs],
        axis=1,
    )
    pair_expectation = np.einsum("rps,sk->rpk", values, pair_bits)
    connected = pair_expectation - np.stack(
        [
            occupations[:, :, left] * occupations[:, :, right]
            for left, right in pairs
        ],
        axis=2,
    )
    block = np.concatenate([occupations, pair_expectation, connected], axis=2)
    names: list[str] = []
    for probe in range(values.shape[1]):
        names.extend(f"probe_{probe}_occupation_site_{site}" for site in range(6))
        names.extend(
            f"probe_{probe}_pair_{left}_{right}" for left, right in pairs
        )
        names.extend(
            f"probe_{probe}_connected_{left}_{right}" for left, right in pairs
        )
    return block.reshape(len(values), -1), tuple(names)


def build_feature_banks(
    probabilities: np.ndarray,
) -> dict[FeatureBankName, np.ndarray]:
    values = validate_probe_probabilities(probabilities)
    symmetric = probabilities_to_symmetric_modes(values)
    six_indices = np.asarray(
        [
            index
            for probe in range(values.shape[1])
            for index in (3 * probe, 3 * probe + 2)
        ],
        dtype=int,
    )
    occupations = probabilities_to_occupations(values).reshape(len(values), -1)
    full, _ = probabilities_to_full_low_order(values)
    banks: dict[FeatureBankName, np.ndarray] = {
        "six_mode_density_curvature": symmetric[:, six_indices],
        "nine_mode_symmetric": symmetric,
        "all_occupations": occupations,
        "full_one_two_body": full,
    }
    if any(not np.isfinite(matrix).all() for matrix in banks.values()):
        raise RuntimeError("feature extraction produced non-finite values")
    return banks
