from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from transition_forecasting.qrc.temporal_rydberg_chain import (
    SUPPORTED_CONDITIONS,
    Condition,
    TemporalRydbergChainConfig,
    _Precomputed,
    _evolve_step_batch,
    _fresh_states,
    _measure_features,
    _occupation_bits,
    _resolve_probe_steps,
    encode_drives,
    interaction_matrix,
)


@dataclass(frozen=True)
class StaggeredLadderGeometryConfig:
    """Six-atom asymmetric 2x3 ladder with several interaction scales."""

    longitudinal_spacing_um: float = 8.5
    row_spacing_um: float = 9.0
    stagger_fraction: float = 0.35
    bottom_spacing_scale: float = 1.05
    defect_site: int = 4
    defect_dx_um: float = 0.35
    defect_dy_um: float = -0.40

    def validate(self) -> None:
        if self.longitudinal_spacing_um <= 0 or self.row_spacing_um <= 0:
            raise ValueError("ladder spacings must be positive")
        if not 0.0 <= self.stagger_fraction < 1.0:
            raise ValueError("stagger_fraction must lie in [0, 1)")
        if self.bottom_spacing_scale <= 0:
            raise ValueError("bottom_spacing_scale must be positive")
        if not 0 <= self.defect_site < 6:
            raise ValueError("defect_site must index one of six ladder atoms")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def staggered_ladder_positions(
    config: StaggeredLadderGeometryConfig,
) -> np.ndarray:
    """Return centered positions for a staggered asymmetric 2x3 ladder."""
    config.validate()
    longitudinal = float(config.longitudinal_spacing_um)
    bottom_spacing = longitudinal * float(config.bottom_spacing_scale)
    stagger = longitudinal * float(config.stagger_fraction)
    half_row = 0.5 * float(config.row_spacing_um)

    top = np.column_stack(
        [
            np.asarray([0.0, longitudinal, 2.0 * longitudinal]),
            np.repeat(half_row, 3),
        ]
    )
    bottom = np.column_stack(
        [
            stagger
            + np.asarray([0.0, bottom_spacing, 2.0 * bottom_spacing]),
            np.repeat(-half_row, 3),
        ]
    )
    positions = np.vstack([top, bottom]).astype(float)
    positions[config.defect_site, 0] += float(config.defect_dx_um)
    positions[config.defect_site, 1] += float(config.defect_dy_um)
    positions -= positions.mean(axis=0, keepdims=True)

    distances = np.sqrt(
        np.sum(
            (positions[:, None, :] - positions[None, :, :]) ** 2,
            axis=-1,
        )
    )
    np.fill_diagonal(distances, np.inf)
    if not np.isfinite(positions).all() or float(distances.min()) <= 0:
        raise ValueError("ladder geometry contains invalid or repeated positions")
    return positions


def ladder_pair_groups() -> dict[str, tuple[tuple[int, int], ...]]:
    """Return structural pair classes used for ladder observables."""
    return {
        "row": ((0, 1), (1, 2), (3, 4), (4, 5)),
        "rung": ((0, 3), (1, 4), (2, 5)),
        "diagonal": ((1, 3), (2, 4)),
        "long": ((0, 4), (1, 5), (0, 5), (2, 3)),
    }


def _unique_pairs(
    groups: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for group in groups:
        for left, right in group:
            pair = tuple(sorted((int(left), int(right))))
            if pair[0] == pair[1] or pair in pairs:
                continue
            pairs.append(pair)
    return tuple(pairs)


def precompute_ladder(
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float = 1.0,
) -> _Precomputed:
    """Precompute ladder interactions and measurement bit patterns."""
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("the staggered ladder requires exactly six atoms")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")

    positions = staggered_ladder_positions(geometry)
    coupling = interaction_matrix(
        positions,
        reservoir.c6_rad_um6_per_us * float(interaction_scale),
    )
    bits = _occupation_bits(6)
    total = bits.sum(axis=1)
    interaction_energy = np.einsum(
        "si,ij,sj->s",
        bits,
        np.triu(coupling, k=1),
        bits,
    )
    groups = ladder_pair_groups()
    nearest_pairs = _unique_pairs(
        (groups["row"], groups["rung"], groups["diagonal"])
    )
    long_pairs = _unique_pairs((groups["long"],))
    nearest_pair_bits = np.stack(
        [bits[:, left] * bits[:, right] for left, right in nearest_pairs],
        axis=1,
    )
    long_pair_bits = np.stack(
        [bits[:, left] * bits[:, right] for left, right in long_pairs],
        axis=1,
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


def _coupling_summary(
    precomputed: _Precomputed,
) -> dict[str, dict[str, float]]:
    groups = ladder_pair_groups()
    payload: dict[str, dict[str, float]] = {}
    for name, pairs in groups.items():
        values = np.asarray(
            [precomputed.interaction_matrix[left, right] for left, right in pairs],
            dtype=float,
        )
        payload[name] = {
            "minimum_rad_us": float(values.min()),
            "mean_rad_us": float(values.mean()),
            "maximum_rad_us": float(values.max()),
        }
    return payload


def build_temporal_rydberg_ladder_features(
    scaled_windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float = 1.0,
    condition: Condition = "ordered",
) -> tuple[np.ndarray, dict[str, object]]:
    """Evolve level/instability windows on the asymmetric ladder."""
    reservoir.validate()
    geometry.validate()
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

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=interaction_scale,
    )
    delta, omega = encode_drives(
        windows,
        reservoir,
        condition=condition,
    )
    samples, steps, _ = windows.shape
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    reset_each_step = condition == "reset"
    interactions = condition != "interaction_off" and interaction_scale > 0
    states = _fresh_states(samples, precomputed.n_atoms)
    rng = (
        np.random.default_rng(reservoir.shot_seed)
        if reservoir.shots is not None
        else None
    )
    feature_blocks: list[np.ndarray] = []

    for step in range(steps):
        if reset_each_step:
            states = _fresh_states(samples, precomputed.n_atoms)
        states = _evolve_step_batch(
            states,
            omega[:, step],
            delta[:, step],
            reservoir,
            precomputed,
            interactions=interactions,
        )
        if step + 1 in probe_steps:
            feature_blocks.append(
                _measure_features(
                    states,
                    precomputed,
                    shots=reservoir.shots,
                    rng=rng,
                )
            )

    features = np.concatenate(feature_blocks, axis=1)
    metadata = {
        "condition": condition,
        "geometry": "staggered_asymmetric_ladder_2x3",
        "interaction_scale": float(interaction_scale),
        "probe_steps": list(probe_steps),
        "feature_count": int(features.shape[1]),
        "positions_um": precomputed.positions.tolist(),
        "nearest_pairs": [list(pair) for pair in precomputed.nearest_pairs],
        "long_pairs": [list(pair) for pair in precomputed.long_pairs],
        "pair_groups": {
            name: [list(pair) for pair in pairs]
            for name, pairs in ladder_pair_groups().items()
        },
        "coupling_summary": _coupling_summary(precomputed),
        "interaction_matrix_rad_us": precomputed.interaction_matrix.tolist(),
        "total_evolution_time_us": float(
            steps * reservoir.step_duration_us
        ),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }
    return features, metadata
