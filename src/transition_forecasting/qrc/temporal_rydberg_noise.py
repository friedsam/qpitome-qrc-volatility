from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _resolve_probe_steps,
    encode_drives,
    precompute,
)


@dataclass(frozen=True)
class TemporalNoiseSpec:
    """Local noise applied throughout the temporal reservoir evolution.

    ``amplitude_damping_t1_us`` and ``dephasing_t2_us`` are physical time
    constants. ``depolarizing_probability`` is the single-qubit depolarizing
    probability accumulated over the complete input sequence; it is converted
    to a per-substep channel before propagation.
    """

    name: str = "ideal"
    amplitude_damping_t1_us: float | None = None
    dephasing_t2_us: float | None = None
    depolarizing_probability: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("noise specification requires a nonempty name")
        if self.amplitude_damping_t1_us is not None and self.amplitude_damping_t1_us <= 0:
            raise ValueError("amplitude_damping_t1_us must be positive")
        if self.dephasing_t2_us is not None and self.dephasing_t2_us <= 0:
            raise ValueError("dephasing_t2_us must be positive")
        if not 0.0 <= self.depolarizing_probability < 1.0:
            raise ValueError("depolarizing_probability must lie in [0, 1)")

    @property
    def is_ideal(self) -> bool:
        return (
            self.amplitude_damping_t1_us is None
            and self.dephasing_t2_us is None
            and self.depolarizing_probability == 0.0
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=None)
def _embedded_single_qubit_operator(
    n_atoms: int,
    site: int,
    flattened_operator: tuple[complex, ...],
) -> np.ndarray:
    if not 0 <= site < n_atoms:
        raise ValueError("site is outside the register")
    operator = np.asarray(flattened_operator, dtype=complex).reshape(2, 2)
    result = np.array([[1.0 + 0.0j]])
    identity = np.eye(2, dtype=complex)
    for index in range(n_atoms):
        result = np.kron(result, operator if index == site else identity)
    return result


def _embed(n_atoms: int, site: int, operator: np.ndarray) -> np.ndarray:
    return _embedded_single_qubit_operator(
        int(n_atoms),
        int(site),
        tuple(complex(value) for value in np.asarray(operator).reshape(-1)),
    )


def _apply_local_kraus_channel(
    density: np.ndarray,
    single_qubit_kraus: tuple[np.ndarray, ...],
    n_atoms: int,
) -> np.ndarray:
    result = np.asarray(density, dtype=complex)
    for site in range(int(n_atoms)):
        updated = np.zeros_like(result)
        for operator in single_qubit_kraus:
            embedded = _embed(int(n_atoms), int(site), operator)
            updated += np.einsum(
                "ia,sab,jb->sij",
                embedded,
                result,
                embedded.conj(),
                optimize=True,
            )
        result = updated
    return result


def _amplitude_damping_kraus(dt_us: float, t1_us: float) -> tuple[np.ndarray, ...]:
    probability = float(1.0 - np.exp(-float(dt_us) / float(t1_us)))
    return (
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - probability)]], dtype=complex),
        np.array([[0.0, np.sqrt(probability)], [0.0, 0.0]], dtype=complex),
    )


def _dephasing_kraus(dt_us: float, t2_us: float) -> tuple[np.ndarray, ...]:
    coherence = float(np.exp(-float(dt_us) / float(t2_us)))
    identity = np.eye(2, dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    return (
        np.sqrt(0.5 * (1.0 + coherence)) * identity,
        np.sqrt(0.5 * (1.0 - coherence)) * z,
    )


def _depolarizing_kraus(
    dt_us: float,
    total_time_us: float,
    total_probability: float,
) -> tuple[np.ndarray, ...]:
    if total_probability <= 0.0:
        return (np.eye(2, dtype=complex),)
    fraction = float(dt_us) / float(total_time_us)
    probability = float(1.0 - (1.0 - float(total_probability)) ** fraction)
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    return (
        np.sqrt(1.0 - 0.75 * probability) * identity,
        np.sqrt(0.25 * probability) * x,
        np.sqrt(0.25 * probability) * y,
        np.sqrt(0.25 * probability) * z,
    )


def _global_rx_matrix(angle: float, n_atoms: int) -> np.ndarray:
    cosine = np.cos(float(angle) / 2.0)
    sine = np.sin(float(angle) / 2.0)
    single = np.array(
        [[cosine, -1.0j * sine], [-1.0j * sine, cosine]],
        dtype=complex,
    )
    result = np.array([[1.0 + 0.0j]])
    for _ in range(int(n_atoms)):
        result = np.kron(result, single)
    return result


def _apply_global_rx_density(
    density: np.ndarray,
    angles: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    output = np.empty_like(density)
    for sample, angle in enumerate(np.asarray(angles, dtype=float)):
        unitary = _global_rx_matrix(float(angle), int(n_atoms))
        output[sample] = unitary @ density[sample] @ unitary.conj().T
    return output


def _apply_noise(
    density: np.ndarray,
    spec: TemporalNoiseSpec,
    *,
    dt_us: float,
    total_time_us: float,
    n_atoms: int,
) -> np.ndarray:
    result = density
    if spec.amplitude_damping_t1_us is not None:
        result = _apply_local_kraus_channel(
            result,
            _amplitude_damping_kraus(dt_us, spec.amplitude_damping_t1_us),
            n_atoms,
        )
    if spec.dephasing_t2_us is not None:
        result = _apply_local_kraus_channel(
            result,
            _dephasing_kraus(dt_us, spec.dephasing_t2_us),
            n_atoms,
        )
    if spec.depolarizing_probability > 0.0:
        result = _apply_local_kraus_channel(
            result,
            _depolarizing_kraus(
                dt_us,
                total_time_us,
                spec.depolarizing_probability,
            ),
            n_atoms,
        )
    return result


def _stabilize_density(density: np.ndarray) -> tuple[np.ndarray, float, float]:
    hermitian = 0.5 * (density + density.conj().transpose(0, 2, 1))
    traces = np.trace(hermitian, axis1=1, axis2=2)
    trace_drift = float(np.max(np.abs(traces - 1.0), initial=0.0))
    if np.any(np.abs(traces) <= 1e-15) or not np.isfinite(traces).all():
        raise RuntimeError("density-matrix trace became invalid")
    hermitian /= traces[:, None, None]
    hermiticity_error = float(
        np.max(
            np.abs(hermitian - hermitian.conj().transpose(0, 2, 1)),
            initial=0.0,
        )
    )
    return hermitian, trace_drift, hermiticity_error


def _measure_probability_features(probabilities: np.ndarray, pre: object) -> np.ndarray:
    probability = np.asarray(probabilities, dtype=float)
    probability = np.clip(probability, 0.0, None)
    probability /= probability.sum(axis=1, keepdims=True)
    occupation = probability @ pre.occupation_bits
    nearest_pair = probability @ pre.nearest_pair_bits
    nearest_connected = nearest_pair - np.stack(
        [
            occupation[:, left] * occupation[:, right]
            for left, right in pre.nearest_pairs
        ],
        axis=1,
    )
    domain_wall = np.stack(
        [
            occupation[:, left]
            + occupation[:, right]
            - 2.0 * nearest_pair[:, edge]
            for edge, (left, right) in enumerate(pre.nearest_pairs)
        ],
        axis=1,
    )
    features = [
        occupation,
        nearest_pair,
        nearest_connected,
        occupation.mean(axis=1, keepdims=True),
        domain_wall.mean(axis=1, keepdims=True),
    ]
    if pre.long_pairs:
        long_pair = probability @ pre.long_pair_bits
        long_connected = long_pair - np.stack(
            [
                occupation[:, left] * occupation[:, right]
                for left, right in pre.long_pairs
            ],
            axis=1,
        )
        features.extend([long_pair, long_connected])
    return np.concatenate(features, axis=1)


def build_noisy_temporal_rydberg_features(
    scaled_windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    noise: TemporalNoiseSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    """Integrate the temporal reservoir with local channels after each substep."""

    reservoir.validate()
    noise.validate()
    if reservoir.shots is not None:
        raise ValueError("density-matrix noise engine requires shots=None")
    windows = np.asarray(scaled_windows, dtype=float)
    if windows.ndim != 3 or windows.shape[2] != 2 or not np.isfinite(windows).all():
        raise ValueError("scaled_windows must have shape (samples, time, 2)")

    pre = precompute(reservoir)
    delta, omega = encode_drives(windows, reservoir, condition="ordered")
    samples, steps, _ = windows.shape
    dimension = 2**reservoir.n_atoms
    density = np.zeros((samples, dimension, dimension), dtype=complex)
    density[:, 0, 0] = 1.0
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    total_time_us = float(steps * reservoir.step_duration_us)
    feature_blocks: list[np.ndarray] = []
    max_trace_drift = 0.0
    max_hermiticity_error = 0.0
    total_substeps = 0

    for step in range(steps):
        scale = max(
            float(np.max(np.abs(omega[:, step]), initial=0.0)),
            float(np.max(np.abs(delta[:, step]), initial=0.0)),
            float(np.max(np.abs(pre.interaction_energy), initial=0.0)),
            1e-12,
        )
        substeps = int(
            np.clip(
                np.ceil(
                    scale
                    * reservoir.step_duration_us
                    / reservoir.max_phase_per_substep
                ),
                1,
                reservoir.max_substeps_per_step,
            )
        )
        total_substeps += substeps
        dt_us = float(reservoir.step_duration_us / substeps)
        diagonal = (
            pre.interaction_energy[None, :]
            - delta[:, step, None] * pre.total_occupation[None, :]
        )
        half_phase = np.exp(-0.5j * dt_us * diagonal)
        full_phase = half_phase * half_phase

        density *= half_phase[:, :, None] * half_phase.conj()[:, None, :]
        for substep in range(substeps):
            density = _apply_global_rx_density(
                density,
                omega[:, step] * dt_us,
                reservoir.n_atoms,
            )
            phase = half_phase if substep == substeps - 1 else full_phase
            density *= phase[:, :, None] * phase.conj()[:, None, :]
            if not noise.is_ideal:
                density = _apply_noise(
                    density,
                    noise,
                    dt_us=dt_us,
                    total_time_us=total_time_us,
                    n_atoms=reservoir.n_atoms,
                )
            density, trace_drift, hermiticity_error = _stabilize_density(density)
            max_trace_drift = max(max_trace_drift, trace_drift)
            max_hermiticity_error = max(max_hermiticity_error, hermiticity_error)

        if step + 1 in probe_steps:
            probabilities = np.real(np.diagonal(density, axis1=1, axis2=2))
            feature_blocks.append(_measure_probability_features(probabilities, pre))

    features = np.concatenate(feature_blocks, axis=1)
    metadata = {
        "noise": noise.to_dict(),
        "probe_steps": list(probe_steps),
        "feature_count": int(features.shape[1]),
        "state_dimension": int(dimension),
        "density_matrix_elements_per_sample": int(dimension * dimension),
        "total_evolution_time_us": total_time_us,
        "total_substeps": int(total_substeps),
        "max_trace_drift_before_renormalization": max_trace_drift,
        "max_hermiticity_error": max_hermiticity_error,
        "reservoir": reservoir.to_dict(),
    }
    return features, metadata
