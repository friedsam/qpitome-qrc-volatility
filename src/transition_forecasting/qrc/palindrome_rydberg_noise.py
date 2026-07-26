"""Density-matrix noise propagation for the frozen A/B/A palindrome reservoir."""
from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.bivariate_crossover_assay import (
    CrossoverSchedule,
    _branch_drive,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    validate_probe_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _resolve_probe_steps,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)
from transition_forecasting.qrc.temporal_rydberg_noise import (
    TemporalNoiseSpec,
    _apply_noise,
    _stabilize_density,
)


def _global_rxy_matrix(angle: float, phase: float, n_atoms: int) -> np.ndarray:
    """Return the global product rotation used by the statevector palindrome."""

    cosine = np.cos(float(angle) / 2.0)
    sine = np.sin(float(angle) / 2.0)
    phase_plus = np.exp(1.0j * float(phase))
    phase_minus = np.conjugate(phase_plus)
    single = np.array(
        [
            [cosine, -1.0j * sine * phase_plus],
            [-1.0j * sine * phase_minus, cosine],
        ],
        dtype=complex,
    )
    result = np.array([[1.0 + 0.0j]])
    for _ in range(int(n_atoms)):
        result = np.kron(result, single)
    return result


def _apply_global_rxy_density(
    density: np.ndarray,
    angles: np.ndarray,
    phases: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Apply sample-specific global XY rotations to batched density matrices."""

    state = np.asarray(density, dtype=complex)
    angle = np.asarray(angles, dtype=float)
    phase = np.asarray(phases, dtype=float)
    dimension = 2 ** int(n_atoms)
    if state.ndim != 3 or state.shape[1:] != (dimension, dimension):
        raise ValueError("density has an incompatible shape")
    if angle.shape != (len(state),) or phase.shape != (len(state),):
        raise ValueError("angles and phases must align with density rows")

    output = np.empty_like(state)
    for sample, (local_angle, local_phase) in enumerate(zip(angle, phase, strict=True)):
        unitary = _global_rxy_matrix(local_angle, local_phase, int(n_atoms))
        output[sample] = unitary @ state[sample] @ unitary.conj().T
    return output


def build_noisy_palindrome_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    schedule: CrossoverSchedule,
    noise: TemporalNoiseSpec,
    *,
    interaction_scale: float,
    drive_phase_rad: float,
    interactions: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Propagate the frozen palindrome with local channels after every substep.

    The ideal-density case is numerically equivalent to
    ``evolve_palindrome_probabilities``. Noise is accumulated in physical time;
    the depolarizing specification is interpreted as a total single-qubit
    probability over the complete input sequence.
    """

    schedule.validate()
    reservoir.validate()
    geometry.validate()
    noise.validate()
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("palindrome density engine requires an exact six-atom reservoir")
    if interaction_scale <= 0:
        raise ValueError("interaction_scale must be positive")
    if not np.isfinite(drive_phase_rad):
        raise ValueError("drive_phase_rad must be finite")

    physical_scale = float(interaction_scale) if interactions else 0.0
    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=physical_scale,
    )
    samples, steps, _ = values.shape
    dimension = 2 ** int(precomputed.n_atoms)
    density = np.zeros((samples, dimension, dimension), dtype=complex)
    density[:, 0, 0] = 1.0
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    total_time_us = float(steps * reservoir.step_duration_us)
    blocks: list[np.ndarray] = []
    max_trace_drift = 0.0
    max_hermiticity_error = 0.0
    total_substeps = 0
    segment_substeps: dict[str, int] = {}

    for step in range(steps):
        for segment_index, (branch, fraction) in enumerate(schedule.segments):
            omega, delta, phase = _branch_drive(
                values,
                step,
                branch,
                reservoir,
                drive_phase_rad,
            )
            duration_us = float(reservoir.step_duration_us * float(fraction))
            scale = max(
                float(np.max(np.abs(omega), initial=0.0)),
                float(np.max(np.abs(delta), initial=0.0)),
                float(np.max(np.abs(precomputed.interaction_energy), initial=0.0)),
                1e-12,
            )
            substeps = int(
                np.clip(
                    np.ceil(scale * duration_us / reservoir.max_phase_per_substep),
                    1,
                    reservoir.max_substeps_per_step,
                )
            )
            total_substeps += substeps
            key = f"segment_{segment_index}_{branch}"
            segment_substeps[key] = segment_substeps.get(key, 0) + substeps
            dt_us = duration_us / float(substeps)
            diagonal = (
                precomputed.interaction_energy[None, :]
                - delta[:, None] * precomputed.total_occupation[None, :]
            )
            half_phase = np.exp(-0.5j * dt_us * diagonal)
            full_phase = half_phase * half_phase

            density *= half_phase[:, :, None] * half_phase.conj()[:, None, :]
            for substep in range(substeps):
                density = _apply_global_rxy_density(
                    density,
                    omega * dt_us,
                    phase,
                    precomputed.n_atoms,
                )
                diagonal_phase = half_phase if substep == substeps - 1 else full_phase
                density *= (
                    diagonal_phase[:, :, None]
                    * diagonal_phase.conj()[:, None, :]
                )
                if not noise.is_ideal:
                    density = _apply_noise(
                        density,
                        noise,
                        dt_us=dt_us,
                        total_time_us=total_time_us,
                        n_atoms=precomputed.n_atoms,
                    )
                density, trace_drift, hermiticity_error = _stabilize_density(density)
                max_trace_drift = max(max_trace_drift, trace_drift)
                max_hermiticity_error = max(
                    max_hermiticity_error,
                    hermiticity_error,
                )

        if step + 1 in probe_steps:
            probabilities = np.real(np.diagonal(density, axis1=1, axis2=2))
            probabilities = np.clip(probabilities, 0.0, None)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1),
        n_atoms=precomputed.n_atoms,
    )
    metadata = {
        "encoding": "symmetric_crossover_density_matrix",
        "schedule": schedule.name,
        "segments": [[branch, float(fraction)] for branch, fraction in schedule.segments],
        "noise": noise.to_dict(),
        "interaction_scale_requested": float(interaction_scale),
        "interaction_scale_applied": physical_scale,
        "interactions_enabled": bool(interactions),
        "drive_phase_rad": float(drive_phase_rad),
        "probe_steps": [int(value) for value in probe_steps],
        "state_dimension": int(dimension),
        "density_matrix_elements_per_sample": int(dimension * dimension),
        "total_evolution_time_us": total_time_us,
        "total_substeps": int(total_substeps),
        "segment_substeps": segment_substeps,
        "max_trace_drift_before_renormalization": float(max_trace_drift),
        "max_hermiticity_error": float(max_hermiticity_error),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }
    return stacked, metadata
