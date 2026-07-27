from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

Condition = Literal[
    "ordered",
    "shuffled",
    "reversed",
    "reset",
    "interaction_off",
    "level_only",
    "rate_only",
]

SUPPORTED_CONDITIONS: tuple[Condition, ...] = (
    "ordered",
    "shuffled",
    "reversed",
    "reset",
    "interaction_off",
    "level_only",
    "rate_only",
)


@dataclass(frozen=True)
class TemporalRydbergChainConfig:
    """Hardware-aligned temporal Rydberg chain used for development assays.

    Input windows are expected to contain two train-fold-scaled channels:
    log-volatility level and its first difference. Level modulates global
    detuning; rate modulates global Rabi amplitude. The state persists across
    the complete window unless the ``reset`` control is requested.
    """

    n_atoms: int = 6
    spacing_short_um: float = 8.5
    spacing_long_um: float = 10.0
    defect_edge: int = 1
    defect_offset_um: float = 0.6
    c6_rad_um6_per_us: float = 5.42e6

    delta_center_rad_us: float = 6.0
    delta_span_rad_us: float = 4.0
    omega_base_rad_us: float = 6.0
    omega_mod_fraction: float = 0.35
    step_duration_us: float = 0.08

    probe_fractions: tuple[float, ...] = (0.5, 1.0)
    max_phase_per_substep: float = 0.25
    max_substeps_per_step: int = 512

    shots: int | None = None
    shot_seed: int = 20260721

    def validate(self) -> None:
        if self.n_atoms < 2:
            raise ValueError("n_atoms must be at least 2")
        if self.spacing_short_um <= 0 or self.spacing_long_um <= 0:
            raise ValueError("chain spacings must be positive")
        if not 0 <= self.defect_edge < self.n_atoms - 1:
            raise ValueError("defect_edge must index an existing chain edge")
        if self.spacing_short_um + min(0.0, self.defect_offset_um) <= 0:
            raise ValueError("defect_offset_um makes a chain spacing non-positive")
        if self.omega_base_rad_us <= 0:
            raise ValueError("omega_base_rad_us must be positive")
        if not 0 <= self.omega_mod_fraction < 1:
            raise ValueError("omega_mod_fraction must lie in [0, 1)")
        if self.step_duration_us <= 0:
            raise ValueError("step_duration_us must be positive")
        if self.max_phase_per_substep <= 0:
            raise ValueError("max_phase_per_substep must be positive")
        if self.max_substeps_per_step < 1:
            raise ValueError("max_substeps_per_step must be positive")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive when provided")
        if not self.probe_fractions:
            raise ValueError("probe_fractions cannot be empty")
        if any(not 0 < value <= 1 for value in self.probe_fractions):
            raise ValueError("probe_fractions must lie in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RobustLevelRateScaler:
    level_median: float
    level_half_range: float
    rate_median: float
    rate_half_range: float
    q_low: float
    q_high: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class _Precomputed:
    positions: np.ndarray
    interaction_matrix: np.ndarray
    occupation_bits: np.ndarray
    total_occupation: np.ndarray
    interaction_energy: np.ndarray
    nearest_pairs: tuple[tuple[int, int], ...]
    nearest_pair_bits: np.ndarray
    long_pairs: tuple[tuple[int, int], ...]
    long_pair_bits: np.ndarray

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])


def alternating_chain_positions(config: TemporalRydbergChainConfig) -> np.ndarray:
    """Return a one-dimensional alternating-spacing chain with one defect edge."""
    config.validate()
    gaps = np.asarray(
        [
            config.spacing_short_um if edge % 2 == 0 else config.spacing_long_um
            for edge in range(config.n_atoms - 1)
        ],
        dtype=float,
    )
    gaps[config.defect_edge] += config.defect_offset_um
    if np.any(gaps <= 0):
        raise ValueError("all chain gaps must remain positive")
    x = np.concatenate(([0.0], np.cumsum(gaps)))
    x -= x.mean()
    return np.column_stack([x, np.zeros_like(x)])


def interaction_matrix(
    positions: np.ndarray,
    c6_rad_um6_per_us: float,
) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (n_atoms, 2)")
    displacement = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(displacement**2, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = c6_rad_um6_per_us / distance**6
    np.fill_diagonal(matrix, 0.0)
    if not np.isfinite(matrix).all():
        raise ValueError("atom positions contain a zero inter-atom distance")
    return matrix


def _long_range_pairs(n_atoms: int) -> tuple[tuple[int, int], ...]:
    candidates = (
        (0, n_atoms - 1),
        (0, n_atoms // 2),
        (n_atoms // 2, n_atoms - 1),
    )
    pairs: list[tuple[int, int]] = []
    for left, right in candidates:
        pair = tuple(sorted((int(left), int(right))))
        if pair[0] == pair[1] or pair[1] - pair[0] <= 1 or pair in pairs:
            continue
        pairs.append(pair)
    return tuple(pairs)


@lru_cache(maxsize=None)
def _occupation_bits(n_atoms: int) -> np.ndarray:
    states = np.arange(2**n_atoms)
    return np.stack(
        [((states >> (n_atoms - 1 - site)) & 1) for site in range(n_atoms)],
        axis=1,
    ).astype(float)


def precompute(config: TemporalRydbergChainConfig) -> _Precomputed:
    config.validate()
    positions = alternating_chain_positions(config)
    coupling = interaction_matrix(positions, config.c6_rad_um6_per_us)
    bits = _occupation_bits(config.n_atoms)
    total = bits.sum(axis=1)
    interaction_energy = np.einsum(
        "si,ij,sj->s",
        bits,
        np.triu(coupling, k=1),
        bits,
    )
    nearest_pairs = tuple(
        (site, site + 1) for site in range(config.n_atoms - 1)
    )
    nearest_pair_bits = np.stack(
        [bits[:, left] * bits[:, right] for left, right in nearest_pairs],
        axis=1,
    )
    long_pairs = _long_range_pairs(config.n_atoms)
    long_pair_bits = (
        np.stack(
            [bits[:, left] * bits[:, right] for left, right in long_pairs],
            axis=1,
        )
        if long_pairs
        else np.zeros((len(bits), 0), dtype=float)
    )
    return _Precomputed(
        positions=positions,
        interaction_matrix=coupling,
        occupation_bits=bits,
        total_occupation=total,
        interaction_energy=interaction_energy,
        nearest_pairs=nearest_pairs,
        nearest_pair_bits=nearest_pair_bits,
        long_pairs=long_pairs,
        long_pair_bits=long_pair_bits,
    )


def _safe_half_range(
    values: np.ndarray,
    q_low: float,
    q_high: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot fit scaler on an empty or non-finite array")
    median = float(np.median(finite))
    low = float(np.quantile(finite, q_low))
    high = float(np.quantile(finite, q_high))
    half = 0.5 * (high - low)
    if not np.isfinite(half) or half <= 1e-12:
        half = float(np.std(finite))
    if not np.isfinite(half) or half <= 1e-12:
        half = 1.0
    return median, half


def _safe_symmetric_scale(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot fit scaler on an empty or non-finite array")
    scale = float(np.quantile(np.abs(finite), quantile))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return scale


def raw_rate_from_level(level_windows: np.ndarray) -> np.ndarray:
    level = np.asarray(level_windows, dtype=float)
    if level.ndim != 2:
        raise ValueError("level_windows must have shape (samples, time)")
    return np.diff(level, axis=1, prepend=level[:, :1])


def fit_level_rate_scaler(
    level_windows: np.ndarray,
    train_mask: np.ndarray,
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> RobustLevelRateScaler:
    level = np.asarray(level_windows, dtype=float)
    mask = np.asarray(train_mask, dtype=bool)
    if level.ndim != 2 or mask.shape != (len(level),):
        raise ValueError("incompatible level_windows and train_mask shapes")
    if not mask.any():
        raise ValueError("train_mask selects no rows")
    if not 0 <= q_low < q_high <= 1:
        raise ValueError("quantiles must satisfy 0 <= q_low < q_high <= 1")
    rate = raw_rate_from_level(level)
    level_median, level_half = _safe_half_range(level[mask], q_low, q_high)
    rate_half = _safe_symmetric_scale(rate[mask], q_high)
    return RobustLevelRateScaler(
        level_median=level_median,
        level_half_range=level_half,
        rate_median=0.0,
        rate_half_range=rate_half,
        q_low=float(q_low),
        q_high=float(q_high),
    )


def transform_level_windows(
    level_windows: np.ndarray,
    scaler: RobustLevelRateScaler,
) -> np.ndarray:
    level = np.asarray(level_windows, dtype=float)
    if level.ndim != 2 or not np.isfinite(level).all():
        raise ValueError("level_windows must be a finite two-dimensional array")
    rate = raw_rate_from_level(level)
    level_scaled = np.clip(
        (level - scaler.level_median) / scaler.level_half_range,
        -1.0,
        1.0,
    )
    rate_scaled = np.clip(
        (rate - scaler.rate_median) / scaler.rate_half_range,
        -1.0,
        1.0,
    )
    return np.stack([level_scaled, rate_scaled], axis=-1)


def controlled_level_windows(
    level_windows: np.ndarray,
    condition: Condition,
    *,
    seed: int,
) -> np.ndarray:
    level = np.asarray(level_windows, dtype=float)
    if level.ndim != 2:
        raise ValueError("level_windows must have shape (samples, time)")
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    if condition == "shuffled":
        rng = np.random.default_rng(seed)
        result = level.copy()
        for sample in range(len(result)):
            result[sample] = level[
                sample,
                rng.permutation(level.shape[1]),
            ]
        return result
    if condition == "reversed":
        return level[:, ::-1].copy()
    return level.copy()


def _apply_global_rx_batch(
    states: np.ndarray,
    angles: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    cosine = np.cos(angles / 2.0)
    sine = np.sin(angles / 2.0)
    result = states
    for qubit in range(n_atoms):
        left = 2**qubit
        right = 2 ** (n_atoms - 1 - qubit)
        view = result.reshape(-1, left, 2, right)
        state_zero = view[:, :, 0, :].copy()
        state_one = view[:, :, 1, :].copy()
        c = cosine[:, None, None]
        s = sine[:, None, None]
        view[:, :, 0, :] = c * state_zero - 1j * s * state_one
        view[:, :, 1, :] = -1j * s * state_zero + c * state_one
        result = view.reshape(-1, 2**n_atoms)
    return result


def _fresh_states(samples: int, n_atoms: int) -> np.ndarray:
    states = np.zeros((samples, 2**n_atoms), dtype=complex)
    states[:, 0] = 1.0
    return states


def _evolve_step_batch(
    states: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
    config: TemporalRydbergChainConfig,
    pre: _Precomputed,
    *,
    interactions: bool,
) -> np.ndarray:
    interaction_energy = (
        pre.interaction_energy
        if interactions
        else np.zeros_like(pre.interaction_energy)
    )
    scale = max(
        float(np.max(np.abs(omega), initial=0.0)),
        float(np.max(np.abs(delta), initial=0.0)),
        float(np.max(np.abs(interaction_energy), initial=0.0)),
        1e-12,
    )
    substeps = int(
        np.ceil(
            scale
            * config.step_duration_us
            / config.max_phase_per_substep
        )
    )
    substeps = int(
        np.clip(substeps, 1, config.max_substeps_per_step)
    )
    dt = config.step_duration_us / substeps
    diagonal = (
        interaction_energy[None, :]
        - delta[:, None] * pre.total_occupation[None, :]
    )
    half = np.exp(-0.5j * dt * diagonal)
    full = half * half
    angles = omega * dt

    result = states * half
    for substep in range(substeps):
        result = _apply_global_rx_batch(result, angles, pre.n_atoms)
        result *= half if substep == substeps - 1 else full
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise RuntimeError("state normalization failed")
    return result / norm


def _resolve_probe_steps(
    length: int,
    fractions: tuple[float, ...],
) -> tuple[int, ...]:
    steps = {
        min(length, max(1, int(round(length * fraction))))
        for fraction in fractions
    }
    steps.add(length)
    return tuple(sorted(steps))


def _measure_features(
    states: np.ndarray,
    pre: _Precomputed,
    *,
    shots: int | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    probabilities = np.abs(states) ** 2
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if shots is not None:
        if rng is None:
            raise ValueError("rng is required for finite-shot features")
        sampled = np.empty_like(probabilities)
        for row, probs in enumerate(probabilities):
            sampled[row] = (
                rng.multinomial(shots, probs) / float(shots)
            )
        probabilities = sampled

    occupation = probabilities @ pre.occupation_bits
    nearest_pair = probabilities @ pre.nearest_pair_bits
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
    density = occupation.mean(axis=1, keepdims=True)
    mean_domain_wall = domain_wall.mean(axis=1, keepdims=True)

    features = [
        occupation,
        nearest_pair,
        nearest_connected,
        density,
        mean_domain_wall,
    ]
    if pre.long_pairs:
        long_pair = probabilities @ pre.long_pair_bits
        long_connected = long_pair - np.stack(
            [
                occupation[:, left] * occupation[:, right]
                for left, right in pre.long_pairs
            ],
            axis=1,
        )
        features.extend([long_pair, long_connected])
    return np.concatenate(features, axis=1)


def encode_drives(
    scaled_windows: np.ndarray,
    config: TemporalRydbergChainConfig,
    *,
    condition: Condition,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.asarray(scaled_windows, dtype=float)
    if windows.ndim != 3 or windows.shape[2] != 2:
        raise ValueError(
            "scaled_windows must have shape (samples, time, 2)"
        )
    level = windows[:, :, 0].copy()
    rate = windows[:, :, 1].copy()
    if condition == "level_only":
        rate.fill(0.0)
    elif condition == "rate_only":
        level.fill(0.0)

    delta = (
        config.delta_center_rad_us
        + config.delta_span_rad_us * level
    )
    omega = config.omega_base_rad_us * (
        1.0 + config.omega_mod_fraction * rate
    )
    if np.any(omega <= 0):
        raise ValueError("encoded Rabi amplitudes must remain positive")
    return delta, omega


def build_temporal_rydberg_chain_features(
    scaled_windows: np.ndarray,
    config: TemporalRydbergChainConfig,
    *,
    condition: Condition = "ordered",
) -> tuple[np.ndarray, dict[str, object]]:
    """Evolve a batch of level/rate windows and return probe observables."""
    config.validate()
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    windows = np.asarray(scaled_windows, dtype=float)
    if (
        windows.ndim != 3
        or windows.shape[2] != 2
        or not np.isfinite(windows).all()
    ):
        raise ValueError(
            "scaled_windows must be finite with shape (samples, time, 2)"
        )

    pre = precompute(config)
    delta, omega = encode_drives(
        windows,
        config,
        condition=condition,
    )
    samples, steps, _ = windows.shape
    probe_steps = _resolve_probe_steps(
        steps,
        config.probe_fractions,
    )
    reset_each_step = condition == "reset"
    interactions = condition != "interaction_off"
    states = _fresh_states(samples, pre.n_atoms)
    rng = (
        np.random.default_rng(config.shot_seed)
        if config.shots is not None
        else None
    )
    feature_blocks: list[np.ndarray] = []

    for step in range(steps):
        if reset_each_step:
            states = _fresh_states(samples, pre.n_atoms)
        states = _evolve_step_batch(
            states,
            omega[:, step],
            delta[:, step],
            config,
            pre,
            interactions=interactions,
        )
        if step + 1 in probe_steps:
            feature_blocks.append(
                _measure_features(
                    states,
                    pre,
                    shots=config.shots,
                    rng=rng,
                )
            )

    features = np.concatenate(feature_blocks, axis=1)
    metadata = {
        "condition": condition,
        "probe_steps": list(probe_steps),
        "feature_count": int(features.shape[1]),
        "positions_um": pre.positions.tolist(),
        "nearest_pairs": [list(pair) for pair in pre.nearest_pairs],
        "long_pairs": [list(pair) for pair in pre.long_pairs],
        "total_evolution_time_us": float(
            steps * config.step_duration_us
        ),
        "config": config.to_dict(),
    }
    return features, metadata


def effective_rank(features: np.ndarray) -> float:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 2:
        return 0.0
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular**2
    total = float(power.sum())
    if total <= 0:
        return 0.0
    probabilities = power[power > 0] / total
    return float(
        np.exp(-np.sum(probabilities * np.log(probabilities)))
    )
