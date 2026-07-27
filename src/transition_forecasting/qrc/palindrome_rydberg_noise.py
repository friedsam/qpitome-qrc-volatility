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
    _amplitude_damping_kraus,
    _dephasing_kraus,
    _depolarizing_kraus,
    _stabilize_density,
)


def _apply_global_rxy_density(
    density: np.ndarray,
    angles: np.ndarray,
    phases: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Apply sample-specific product rotations without dense 64x64 unitaries."""

    result = np.asarray(density, dtype=complex).copy()
    angle = np.asarray(angles, dtype=float)
    phase = np.asarray(phases, dtype=float)
    samples = len(result)
    dimension = 2 ** int(n_atoms)
    if result.ndim != 3 or result.shape[1:] != (dimension, dimension):
        raise ValueError("density has an incompatible shape")
    if angle.shape != (samples,) or phase.shape != (samples,):
        raise ValueError("angles and phases must align with density rows")

    cosine = np.cos(angle / 2.0)
    sine = np.sin(angle / 2.0)
    phase_plus = np.exp(1.0j * phase)
    phase_minus = np.conjugate(phase_plus)
    c = cosine[:, None, None, None]
    s = sine[:, None, None, None]
    plus = phase_plus[:, None, None, None]
    minus = phase_minus[:, None, None, None]

    # Left multiplication by U = tensor_product(R_xy).
    for site in range(int(n_atoms)):
        left = 2**site
        right = 2 ** (int(n_atoms) - 1 - site)
        view = result.reshape(samples, left, 2, right, dimension)
        row_zero = view[:, :, 0, :, :].copy()
        row_one = view[:, :, 1, :, :].copy()
        view[:, :, 0, :, :] = c * row_zero - 1.0j * s * plus * row_one
        view[:, :, 1, :, :] = -1.0j * s * minus * row_zero + c * row_one
        result = view.reshape(samples, dimension, dimension)

    # Right multiplication by U dagger, equivalently U* on the bra indices.
    for site in range(int(n_atoms)):
        left = 2**site
        right = 2 ** (int(n_atoms) - 1 - site)
        view = result.reshape(samples, dimension, left, 2, right)
        column_zero = view[:, :, :, 0, :].copy()
        column_one = view[:, :, :, 1, :].copy()
        view[:, :, :, 0, :] = c * column_zero + 1.0j * s * minus * column_one
        view[:, :, :, 1, :] = 1.0j * s * plus * column_zero + c * column_one
        result = view.reshape(samples, dimension, dimension)
    return result


def _apply_single_site_kraus_term(
    density: np.ndarray,
    operator: np.ndarray,
    *,
    n_atoms: int,
    site: int,
) -> np.ndarray:
    """Apply K rho K dagger to one site using tensor reshapes."""

    result = np.asarray(density, dtype=complex).copy()
    matrix = np.asarray(operator, dtype=complex)
    samples = len(result)
    dimension = 2 ** int(n_atoms)
    if result.shape != (samples, dimension, dimension):
        raise ValueError("density has an incompatible shape")
    if matrix.shape != (2, 2):
        raise ValueError("single-site Kraus operator must be 2x2")
    if not 0 <= int(site) < int(n_atoms):
        raise ValueError("site is outside the register")

    left = 2 ** int(site)
    right = 2 ** (int(n_atoms) - 1 - int(site))
    view = result.reshape(samples, left, 2, right, dimension)
    row_zero = view[:, :, 0, :, :].copy()
    row_one = view[:, :, 1, :, :].copy()
    view[:, :, 0, :, :] = matrix[0, 0] * row_zero + matrix[0, 1] * row_one
    view[:, :, 1, :, :] = matrix[1, 0] * row_zero + matrix[1, 1] * row_one
    result = view.reshape(samples, dimension, dimension)

    view = result.reshape(samples, dimension, left, 2, right)
    column_zero = view[:, :, :, 0, :].copy()
    column_one = view[:, :, :, 1, :].copy()
    view[:, :, :, 0, :] = (
        np.conjugate(matrix[0, 0]) * column_zero
        + np.conjugate(matrix[0, 1]) * column_one
    )
    view[:, :, :, 1, :] = (
        np.conjugate(matrix[1, 0]) * column_zero
        + np.conjugate(matrix[1, 1]) * column_one
    )
    return view.reshape(samples, dimension, dimension)


def _apply_local_kraus_channel_fast(
    density: np.ndarray,
    single_qubit_kraus: tuple[np.ndarray, ...],
    n_atoms: int,
) -> np.ndarray:
    """Apply the same local channel independently to every site."""

    result = np.asarray(density, dtype=complex)
    for site in range(int(n_atoms)):
        updated = np.zeros_like(result)
        for operator in single_qubit_kraus:
            updated += _apply_single_site_kraus_term(
                result,
                operator,
                n_atoms=int(n_atoms),
                site=site,
            )
        result = updated
    return result


def _apply_noise_fast(
    density: np.ndarray,
    spec: TemporalNoiseSpec,
    *,
    dt_us: float,
    total_time_us: float,
    n_atoms: int,
) -> np.ndarray:
    result = density
    if spec.amplitude_damping_t1_us is not None:
        result = _apply_local_kraus_channel_fast(
            result,
            _amplitude_damping_kraus(dt_us, spec.amplitude_damping_t1_us),
            n_atoms,
        )
    if spec.dephasing_t2_us is not None:
        result = _apply_local_kraus_channel_fast(
            result,
            _dephasing_kraus(dt_us, spec.dephasing_t2_us),
            n_atoms,
        )
    if spec.depolarizing_probability > 0.0:
        result = _apply_local_kraus_channel_fast(
            result,
            _depolarizing_kraus(
                dt_us,
                total_time_us,
                spec.depolarizing_probability,
            ),
            n_atoms,
        )
    return result


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
                    density = _apply_noise_fast(
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
        "implementation": "tensorized_local_density_updates",
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
