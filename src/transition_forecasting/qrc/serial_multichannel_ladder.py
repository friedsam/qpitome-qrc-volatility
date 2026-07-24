from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    validate_probe_probabilities,
)
from transition_forecasting.qrc.representation_candidates import (
    local_instability,
    raw_rate,
    slope_disagreement,
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

MultichannelRepresentation = Literal[
    "level_rate_time",
    "level_instability_slope",
]
MULTICHANNEL_REPRESENTATIONS: tuple[MultichannelRepresentation, ...] = (
    "level_rate_time",
    "level_instability_slope",
)
MultichannelCondition = Literal["ordered", "shuffled", "interaction_off"]
MULTICHANNEL_CONDITIONS: tuple[MultichannelCondition, ...] = (
    "ordered",
    "shuffled",
    "interaction_off",
)


@dataclass(frozen=True)
class ThreeChannelScaler:
    medians: tuple[float, float, float]
    half_ranges: tuple[float, float, float]
    q_low: float
    q_high: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SerialMultichannelEncodingConfig:
    """Three standard-control pulses per market day on the frozen ladder.

    Channel 0 modulates global detuning at phase 0. Channel 1 modulates
    global Rabi amplitude at phase pi/2. Channel 2 modulates global detuning
    at phase pi. The three pulse durations sum to the original per-day step
    duration, so the total physical evolution time remains unchanged.
    """

    q_low: float = 0.01
    q_high: float = 0.99
    instability_window: int = 5
    short_slope_window: int = 5
    long_slope_window: int = 20
    channel_phases_rad: tuple[float, float, float] = (
        0.0,
        0.5 * np.pi,
        np.pi,
    )
    pulse_duration_fractions: tuple[float, float, float] = (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )
    third_channel_delta_span_rad_us: float = 3.0
    shuffle_seed: int = 20260724

    def validate(self) -> None:
        if not 0.0 <= self.q_low < self.q_high <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= q_low < q_high <= 1")
        if self.instability_window < 1:
            raise ValueError("instability_window must be positive")
        if self.short_slope_window < 2:
            raise ValueError("short_slope_window must be at least two")
        if self.long_slope_window < self.short_slope_window:
            raise ValueError("long_slope_window must be at least short_slope_window")
        if len(self.channel_phases_rad) != 3 or not np.isfinite(
            self.channel_phases_rad
        ).all():
            raise ValueError("exactly three finite channel phases are required")
        fractions = np.asarray(self.pulse_duration_fractions, dtype=float)
        if fractions.shape != (3,) or np.any(fractions <= 0):
            raise ValueError("exactly three positive pulse fractions are required")
        if not np.isclose(float(fractions.sum()), 1.0, atol=1e-12):
            raise ValueError("pulse duration fractions must sum to one")
        if self.third_channel_delta_span_rad_us <= 0:
            raise ValueError("third-channel detuning span must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_multichannel_sequences(
    level_windows: np.ndarray,
    representation: MultichannelRepresentation,
    config: SerialMultichannelEncodingConfig,
) -> np.ndarray:
    config.validate()
    level = np.asarray(level_windows, dtype=float)
    if level.ndim != 2 or not np.isfinite(level).all():
        raise ValueError("level_windows must be finite with shape (samples, time)")
    if representation not in MULTICHANNEL_REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {representation}")

    if representation == "level_rate_time":
        second = raw_rate(level)
        clock = np.linspace(-1.0, 1.0, level.shape[1], dtype=float)
        third = np.broadcast_to(clock[None, :], level.shape).copy()
    else:
        second = local_instability(level, config.instability_window)
        third = slope_disagreement(
            level,
            short_window=config.short_slope_window,
            long_window=config.long_slope_window,
        )
    return np.stack([level, second, third], axis=-1)


def _safe_center_scale(
    values: np.ndarray,
    q_low: float,
    q_high: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot scale an empty channel")
    median = float(np.median(finite))
    low = float(np.quantile(finite, q_low))
    high = float(np.quantile(finite, q_high))
    half = 0.5 * (high - low)
    if not np.isfinite(half) or half <= 1e-12:
        half = float(np.std(finite))
    if not np.isfinite(half) or half <= 1e-12:
        half = 1.0
    return median, half


def fit_three_channel_scaler(
    sequences: np.ndarray,
    train_mask: np.ndarray,
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> ThreeChannelScaler:
    values = np.asarray(sequences, dtype=float)
    train = np.asarray(train_mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("sequences must have shape (samples, time, 3)")
    if train.shape != (len(values),) or not train.any():
        raise ValueError("train_mask must select aligned rows")
    centers = [
        _safe_center_scale(values[train, :, channel], q_low, q_high)
        for channel in range(3)
    ]
    return ThreeChannelScaler(
        medians=tuple(float(center) for center, _ in centers),
        half_ranges=tuple(float(scale) for _, scale in centers),
        q_low=float(q_low),
        q_high=float(q_high),
    )


def transform_three_channel_sequences(
    sequences: np.ndarray,
    scaler: ThreeChannelScaler,
) -> np.ndarray:
    values = np.asarray(sequences, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3 or not np.isfinite(values).all():
        raise ValueError("sequences must be finite with shape (samples, time, 3)")
    center = np.asarray(scaler.medians, dtype=float)[None, None, :]
    scale = np.asarray(scaler.half_ranges, dtype=float)[None, None, :]
    return np.clip((values - center) / scale, -1.0, 1.0)


def controlled_multichannel_sequences(
    scaled_sequences: np.ndarray,
    representation: MultichannelRepresentation,
    condition: MultichannelCondition,
    *,
    seed: int,
) -> np.ndarray:
    values = np.asarray(scaled_sequences, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("scaled_sequences must have shape (samples, time, 3)")
    if representation not in MULTICHANNEL_REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {representation}")
    if condition not in MULTICHANNEL_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    if condition != "shuffled":
        return values.copy()

    rng = np.random.default_rng(int(seed))
    result = values.copy()
    for sample in range(len(values)):
        permutation = rng.permutation(values.shape[1])
        if representation == "level_rate_time":
            # Preserve the explicit clock while destroying the ordering of market inputs.
            result[sample, :, :2] = values[sample, permutation, :2]
        else:
            result[sample] = values[sample, permutation]
    return result


def _apply_global_xy_batch(
    states: np.ndarray,
    angles: np.ndarray,
    phases: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Apply uniform rotations generated by cos(phi)X - sin(phi)Y."""

    theta = np.asarray(angles, dtype=float)
    phi = np.asarray(phases, dtype=float)
    if theta.shape != (len(states),) or phi.shape != (len(states),):
        raise ValueError("angles and phases must align with state rows")
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    phase_up = np.exp(1j * phi)
    phase_down = np.exp(-1j * phi)
    result = states
    for qubit in range(n_atoms):
        left = 2**qubit
        right = 2 ** (n_atoms - 1 - qubit)
        view = result.reshape(-1, left, 2, right)
        state_zero = view[:, :, 0, :].copy()
        state_one = view[:, :, 1, :].copy()
        c = cosine[:, None, None]
        s = sine[:, None, None]
        view[:, :, 0, :] = c * state_zero - 1j * s * phase_up[:, None, None] * state_one
        view[:, :, 1, :] = -1j * s * phase_down[:, None, None] * state_zero + c * state_one
        result = view.reshape(-1, 2**n_atoms)
    return result


def _evolve_control_pulse_batch(
    states: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
    phase: np.ndarray,
    *,
    duration_us: float,
    reservoir: TemporalRydbergChainConfig,
    precomputed: object,
    interactions: bool,
) -> np.ndarray:
    interaction_energy = (
        precomputed.interaction_energy
        if interactions
        else np.zeros_like(precomputed.interaction_energy)
    )
    scale = max(
        float(np.max(np.abs(omega), initial=0.0)),
        float(np.max(np.abs(delta), initial=0.0)),
        float(np.max(np.abs(interaction_energy), initial=0.0)),
        1e-12,
    )
    substeps = int(
        np.ceil(scale * float(duration_us) / reservoir.max_phase_per_substep)
    )
    substeps = int(np.clip(substeps, 1, reservoir.max_substeps_per_step))
    dt = float(duration_us) / substeps
    diagonal = interaction_energy[None, :] - delta[:, None] * precomputed.total_occupation[None, :]
    half = np.exp(-0.5j * dt * diagonal)
    full = half * half
    angles = omega * dt

    result = states * half
    for substep in range(substeps):
        result = _apply_global_xy_batch(
            result,
            angles,
            phase,
            precomputed.n_atoms,
        )
        result *= half if substep == substeps - 1 else full
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise RuntimeError("state normalization failed")
    return result / norm


def _channel_controls(
    channel: int,
    values: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    encoding: SerialMultichannelEncodingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = len(values)
    phase = np.full(samples, float(encoding.channel_phases_rad[channel]))
    if channel == 0:
        delta = reservoir.delta_center_rad_us + reservoir.delta_span_rad_us * values
        omega = np.full(samples, reservoir.omega_base_rad_us)
    elif channel == 1:
        delta = np.full(samples, reservoir.delta_center_rad_us)
        omega = reservoir.omega_base_rad_us * (
            1.0 + reservoir.omega_mod_fraction * values
        )
    elif channel == 2:
        delta = (
            reservoir.delta_center_rad_us
            + encoding.third_channel_delta_span_rad_us * values
        )
        omega = np.full(samples, reservoir.omega_base_rad_us)
    else:
        raise ValueError("channel must be 0, 1, or 2")
    if np.any(omega <= 0):
        raise ValueError("encoded Rabi amplitudes must remain positive")
    return delta, omega, phase


def evolve_serial_ladder_probe_probabilities(
    scaled_sequences: np.ndarray,
    representation: MultichannelRepresentation,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    encoding: SerialMultichannelEncodingConfig,
    *,
    interaction_scale: float = 1.25,
    condition: MultichannelCondition = "ordered",
) -> tuple[np.ndarray, dict[str, object]]:
    reservoir.validate()
    geometry.validate()
    encoding.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("serial ladder evolution requires an exact six-atom reservoir")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")
    if representation not in MULTICHANNEL_REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {representation}")
    if condition not in MULTICHANNEL_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")

    windows = controlled_multichannel_sequences(
        scaled_sequences,
        representation,
        condition,
        seed=encoding.shuffle_seed,
    )
    if not np.isfinite(windows).all():
        raise ValueError("scaled sequences contain non-finite values")
    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    samples, days, _ = windows.shape
    probe_days = _resolve_probe_steps(days, reservoir.probe_fractions)
    states = _fresh_states(samples, precomputed.n_atoms)
    interactions = condition != "interaction_off" and interaction_scale > 0
    pulse_fractions = np.asarray(encoding.pulse_duration_fractions, dtype=float)
    blocks: list[np.ndarray] = []

    for day in range(days):
        for channel in range(3):
            delta, omega, phase = _channel_controls(
                channel,
                windows[:, day, channel],
                reservoir,
                encoding,
            )
            states = _evolve_control_pulse_batch(
                states,
                omega,
                delta,
                phase,
                duration_us=reservoir.step_duration_us * pulse_fractions[channel],
                reservoir=reservoir,
                precomputed=precomputed,
                interactions=interactions,
            )
        if day + 1 in probe_days:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1),
        n_atoms=precomputed.n_atoms,
    )
    metadata: dict[str, object] = {
        "representation": representation,
        "condition": condition,
        "geometry": "staggered_asymmetric_ladder_2x3",
        "interaction_scale": float(interaction_scale),
        "probe_steps": [int(value) for value in probe_days],
        "samples": int(samples),
        "states": int(2**precomputed.n_atoms),
        "market_days": int(days),
        "pulses_per_day": 3,
        "total_evolution_time_us": float(days * reservoir.step_duration_us),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
        "encoding_config": encoding.to_dict(),
        "positions_um": precomputed.positions.tolist(),
    }
    return stacked, metadata
