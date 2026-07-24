"""Position-encoded static Rydberg reservoir for image classification.

For N atoms, N-1 input features set the nearest-neighbour spacings. The data
therefore enter simultaneously through V_ij=C6/r_ij^6 while the drive remains
global. Qubit count and input dimension co-vary under this encoding and must be
reported together.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from transition_forecasting.qrc.temporal_rydberg_chain import (
    _Precomputed,
    _apply_global_rx_batch,
    _fresh_states,
    _measure_features,
    _occupation_bits,
    interaction_matrix,
)

C6_RAD_UM6_PER_US = 5.42e6


@dataclass(frozen=True)
class PositionEncodedConfig:
    n_atoms: int = 10
    spacing_min_um: float = 8.0
    spacing_max_um: float = 12.0
    c6_rad_um6_per_us: float = C6_RAD_UM6_PER_US
    omega_rad_us: float = 6.0
    delta_rad_us: float = 6.0
    total_time_us: float = 1.6
    probe_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    max_phase_per_substep: float = 0.25
    max_substeps: int = 20000
    shots: int | None = None
    shot_seed: int = 20260722

    def validate(self) -> None:
        if self.n_atoms < 3:
            raise ValueError("n_atoms must be at least three")
        if not 0 < self.spacing_min_um < self.spacing_max_um:
            raise ValueError("require 0 < spacing_min_um < spacing_max_um")
        if self.spacing_min_um < 4.0:
            raise ValueError("spacing_min_um is below the Aquila minimum")
        if self.omega_rad_us <= 0 or self.total_time_us <= 0:
            raise ValueError("omega_rad_us and total_time_us must be positive")
        if not self.probe_fractions or any(
            not 0 < value <= 1 for value in self.probe_fractions
        ):
            raise ValueError("probe_fractions must lie in (0, 1]")
        if self.max_phase_per_substep <= 0 or self.max_substeps < 1:
            raise ValueError("invalid integration controls")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive when supplied")

    @property
    def feature_dim(self) -> int:
        return self.n_atoms - 1

    def blockade_radius_um(self) -> float:
        return float(
            (self.c6_rad_um6_per_us / self.omega_rad_us) ** (1.0 / 6.0)
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_dim"] = self.feature_dim
        payload["blockade_radius_um"] = self.blockade_radius_um()
        return payload


def encode_positions(
    unit_features: np.ndarray, config: PositionEncodedConfig
) -> np.ndarray:
    """Map [0,1] features to gaps; larger values produce stronger coupling."""
    config.validate()
    values = np.asarray(unit_features, dtype=float)
    if values.ndim != 2 or values.shape[1] != config.feature_dim:
        raise ValueError(
            f"unit_features must have shape (samples, {config.feature_dim})"
        )
    if not np.isfinite(values).all() or values.min() < -1e-9 or values.max() > 1 + 1e-9:
        raise ValueError("unit_features must be finite and lie in [0,1]")
    span = config.spacing_max_um - config.spacing_min_um
    gaps = config.spacing_max_um - span * np.clip(values, 0.0, 1.0)
    x = np.concatenate(
        [np.zeros((len(gaps), 1)), np.cumsum(gaps, axis=1)], axis=1
    )
    x -= x.mean(axis=1, keepdims=True)
    return np.stack([x, np.zeros_like(x)], axis=-1)


def sample_interaction_energies(
    positions: np.ndarray, config: PositionEncodedConfig
) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (config.n_atoms, 2):
        raise ValueError("positions must have shape (samples, n_atoms, 2)")
    bits = _occupation_bits(config.n_atoms)
    couplings = np.stack(
        [
            np.triu(
                interaction_matrix(sample, config.c6_rad_um6_per_us), k=1
            )
            for sample in points
        ]
    )
    return np.einsum("bi,nij,bj->nb", bits, couplings, bits)


def _long_pairs(n_atoms: int) -> tuple[tuple[int, int], ...]:
    candidates = (
        (0, n_atoms - 1),
        (0, n_atoms // 2),
        (n_atoms // 2, n_atoms - 1),
    )
    output: list[tuple[int, int]] = []
    for left, right in candidates:
        pair = tuple(sorted((left, right)))
        if pair[1] - pair[0] > 1 and pair not in output:
            output.append(pair)
    return tuple(output)


def _static_precompute(config: PositionEncodedConfig) -> _Precomputed:
    bits = _occupation_bits(config.n_atoms)
    nearest = tuple((site, site + 1) for site in range(config.n_atoms - 1))
    nearest_bits = np.stack(
        [bits[:, left] * bits[:, right] for left, right in nearest], axis=1
    )
    long_pairs = _long_pairs(config.n_atoms)
    long_bits = (
        np.stack([bits[:, left] * bits[:, right] for left, right in long_pairs], axis=1)
        if long_pairs
        else np.zeros((len(bits), 0), dtype=float)
    )
    return _Precomputed(
        positions=np.zeros((config.n_atoms, 2), dtype=float),
        interaction_matrix=np.zeros((config.n_atoms, config.n_atoms), dtype=float),
        occupation_bits=bits,
        total_occupation=bits.sum(axis=1),
        interaction_energy=np.zeros(len(bits), dtype=float),
        nearest_pairs=nearest,
        nearest_pair_bits=nearest_bits,
        long_pairs=long_pairs,
        long_pair_bits=long_bits,
    )


def build_position_encoded_features(
    unit_features: np.ndarray,
    config: PositionEncodedConfig,
    *,
    interactions: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evolve a batch under constant global drive and return probe features."""
    config.validate()
    values = np.asarray(unit_features, dtype=float)
    pre = _static_precompute(config)
    positions = encode_positions(values, config)
    energies = (
        sample_interaction_energies(positions, config)
        if interactions
        else np.zeros((len(values), 2**config.n_atoms), dtype=float)
    )
    scale = max(
        config.omega_rad_us,
        abs(config.delta_rad_us),
        float(np.max(np.abs(energies), initial=0.0)),
        1e-12,
    )
    substeps = int(
        np.clip(
            np.ceil(
                scale
                * config.total_time_us
                / config.max_phase_per_substep
            ),
            1,
            config.max_substeps,
        )
    )
    probe_steps = sorted(
        {
            min(substeps, max(1, int(round(substeps * fraction))))
            for fraction in config.probe_fractions
        }
        | {substeps}
    )
    dt = config.total_time_us / substeps
    diagonal = energies - config.delta_rad_us * pre.total_occupation[None, :]
    half = np.exp(-0.5j * dt * diagonal)
    full = half * half
    angles = np.full(len(values), config.omega_rad_us * dt)
    rng = (
        np.random.default_rng(config.shot_seed)
        if config.shots is not None
        else None
    )
    states = _fresh_states(len(values), config.n_atoms) * half
    blocks: list[np.ndarray] = []
    for step in range(1, substeps + 1):
        states = _apply_global_rx_batch(states, angles, config.n_atoms)
        states *= half if step == substeps else full
        if step in probe_steps:
            norm = np.linalg.norm(states, axis=1, keepdims=True)
            if np.any(norm <= 0) or not np.isfinite(norm).all():
                raise RuntimeError("state normalization failed")
            blocks.append(
                _measure_features(
                    states / norm,
                    pre,
                    shots=config.shots,
                    rng=rng,
                )
            )
    features = np.concatenate(blocks, axis=1)
    metadata = {
        "encoding": "position",
        "interactions": bool(interactions),
        "n_atoms": config.n_atoms,
        "feature_dim": config.feature_dim,
        "feature_count": int(features.shape[1]),
        "substeps": substeps,
        "probe_steps": probe_steps,
        "blockade_radius_um": config.blockade_radius_um(),
        "spacing_range_um": [config.spacing_min_um, config.spacing_max_um],
        "config": config.to_dict(),
    }
    return features, metadata


@dataclass(frozen=True)
class UnitScaler:
    minimum: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.asarray(values, dtype=float) - self.minimum) / self.scale,
            0.0,
            1.0,
        )


def fit_unit_scaler(train_features: np.ndarray) -> UnitScaler:
    matrix = np.asarray(train_features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("train_features must be two-dimensional")
    minimum = matrix.min(axis=0)
    scale = matrix.max(axis=0) - minimum
    return UnitScaler(minimum=minimum, scale=np.where(scale > 1e-12, scale, 1.0))
