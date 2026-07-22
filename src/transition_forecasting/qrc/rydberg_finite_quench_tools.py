from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from transition_forecasting.qrc.ladder_mode_readout_tools import (
    ladder_mode_weights,
)
from transition_forecasting.qrc.rydberg_susceptibility_tools import (
    PreparedLadder,
    _occupation_estimate,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _evolve_step_batch,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


@dataclass(frozen=True)
class FiniteQuenchAssayConfig:
    """Development-only finite-quench L5 warning protocol."""

    folds: tuple[int, ...] = tuple(range(1, 9))
    lead: int = 5
    max_per_class: int = 12
    sequence_length: int = 40
    prequential_blocks: int = 5
    interaction_scale: float = 1.25
    quench_delta_offset_rad_us: float = 2.0
    response_probe_steps: tuple[int, ...] = (3, 5)
    classifier_c: float = 0.1
    shot_budgets: tuple[int, ...] = (1000, 5000)
    shot_replicates: int = 5
    seed: int = 20260722
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or any(int(fold) < 1 for fold in self.folds):
            raise ValueError("folds must be positive and nonempty")
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be unique")
        if self.lead < 1:
            raise ValueError("lead must be positive")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if self.quench_delta_offset_rad_us <= 0:
            raise ValueError("quench delta offset must be positive")
        if not self.response_probe_steps:
            raise ValueError("response_probe_steps cannot be empty")
        if tuple(sorted(set(self.response_probe_steps))) != self.response_probe_steps:
            raise ValueError("response_probe_steps must be unique and sorted")
        if self.response_probe_steps[0] < 1:
            raise ValueError("response probe steps must be positive")
        if self.classifier_c <= 0:
            raise ValueError("classifier_c must be positive")
        if any(int(shots) < 1 for shots in self.shot_budgets):
            raise ValueError("shot budgets must be positive")
        if self.shot_replicates < 1:
            raise ValueError("shot_replicates must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evolve_finite_quench(
    prepared: PreparedLadder,
    reservoir: TemporalRydbergChainConfig,
    *,
    delta_offset_rad_us: float,
    response_probe_steps: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Continue a history-prepared state under one finite positive detuning quench."""

    if delta_offset_rad_us <= 0:
        raise ValueError("delta_offset_rad_us must be positive")
    if not response_probe_steps:
        raise ValueError("response_probe_steps cannot be empty")
    maximum = max(int(value) for value in response_probe_steps)
    requested = set(int(value) for value in response_probe_steps)
    states = prepared.final_states.copy()
    snapshots: dict[int, np.ndarray] = {}
    for step in range(1, maximum + 1):
        states = _evolve_step_batch(
            states,
            prepared.final_omega,
            prepared.final_delta + float(delta_offset_rad_us),
            reservoir,
            prepared.precomputed,
            interactions=True,
        )
        if step in requested:
            snapshots[int(step)] = states.copy()
    return snapshots


def finite_quench_feature_blocks(
    prepared: PreparedLadder,
    quench_states: dict[int, np.ndarray],
    geometry: StaggeredLadderGeometryConfig,
    *,
    response_probe_steps: tuple[int, ...],
    shots: int | None,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Build raw and baseline-referenced finite-quench reporter features.

    Every derived feature uses only three measurement settings for the default
    protocol: the prepared final state and the post-quench states at steps 3 and 5.
    """

    if set(int(value) for value in response_probe_steps) != set(quench_states):
        raise ValueError("quench states do not match requested response steps")
    rng = None if shots is None else np.random.default_rng(int(seed))
    weights = ladder_mode_weights()[:, :3]
    reporter = int(geometry.defect_site)
    bulk = np.asarray([site for site in range(6) if site != reporter], dtype=int)

    final_occ = _occupation_estimate(
        prepared.final_states,
        prepared.precomputed,
        shots=shots,
        rng=rng,
    )
    final_modes = final_occ @ weights
    final_reporter = final_occ[:, reporter]
    final_reporter_minus_bulk = final_reporter - final_occ[:, bulk].mean(axis=1)

    quench_mode_raw: list[np.ndarray] = []
    quench_mode_delta: list[np.ndarray] = []
    reporter_raw: list[np.ndarray] = []
    reporter_delta: list[np.ndarray] = []
    quench_mode_raw_names: list[str] = []
    quench_mode_delta_names: list[str] = []
    reporter_raw_names: list[str] = []
    reporter_delta_names: list[str] = []

    for probe in response_probe_steps:
        occupation = _occupation_estimate(
            quench_states[int(probe)],
            prepared.precomputed,
            shots=shots,
            rng=rng,
        )
        modes = occupation @ weights
        mode_delta = modes - final_modes
        reporter_value = occupation[:, reporter]
        reporter_minus_bulk = reporter_value - occupation[:, bulk].mean(axis=1)

        quench_mode_raw.append(modes)
        quench_mode_delta.append(mode_delta)
        reporter_raw.append(
            np.column_stack([reporter_value, reporter_minus_bulk])
        )
        reporter_delta.append(
            np.column_stack(
                [
                    reporter_value - final_reporter,
                    reporter_minus_bulk - final_reporter_minus_bulk,
                ]
            )
        )
        quench_mode_raw_names.extend(
            (
                f"quench_{probe}_symmetric_constant",
                f"quench_{probe}_symmetric_gradient",
                f"quench_{probe}_symmetric_curvature",
            )
        )
        quench_mode_delta_names.extend(
            (
                f"quench_{probe}_delta_symmetric_constant",
                f"quench_{probe}_delta_symmetric_gradient",
                f"quench_{probe}_delta_symmetric_curvature",
            )
        )
        reporter_raw_names.extend(
            (
                f"quench_{probe}_reporter",
                f"quench_{probe}_reporter_minus_bulk",
            )
        )
        reporter_delta_names.extend(
            (
                f"quench_{probe}_delta_reporter",
                f"quench_{probe}_delta_reporter_minus_bulk",
            )
        )

    blocks: dict[str, np.ndarray] = {
        "final_static": final_modes,
        "quench_modes_raw": np.concatenate(quench_mode_raw, axis=1),
        "quench_modes_delta": np.concatenate(quench_mode_delta, axis=1),
        "reporter_raw": np.concatenate(reporter_raw, axis=1),
        "reporter_delta": np.concatenate(reporter_delta, axis=1),
    }
    blocks["quench_modes"] = np.concatenate(
        [blocks["quench_modes_raw"], blocks["quench_modes_delta"]], axis=1
    )
    blocks["reporter_quench"] = np.concatenate(
        [blocks["reporter_raw"], blocks["reporter_delta"]], axis=1
    )
    blocks["finite_quench_all"] = np.concatenate(
        [blocks["quench_modes"], blocks["reporter_quench"]], axis=1
    )
    blocks["qrc_all"] = np.concatenate(
        [blocks["final_static"], blocks["finite_quench_all"]], axis=1
    )

    names: dict[str, tuple[str, ...]] = {
        "final_static": (
            "final_symmetric_constant",
            "final_symmetric_gradient",
            "final_symmetric_curvature",
        ),
        "quench_modes_raw": tuple(quench_mode_raw_names),
        "quench_modes_delta": tuple(quench_mode_delta_names),
        "reporter_raw": tuple(reporter_raw_names),
        "reporter_delta": tuple(reporter_delta_names),
    }
    names["quench_modes"] = (
        names["quench_modes_raw"] + names["quench_modes_delta"]
    )
    names["reporter_quench"] = names["reporter_raw"] + names["reporter_delta"]
    names["finite_quench_all"] = names["quench_modes"] + names["reporter_quench"]
    names["qrc_all"] = names["final_static"] + names["finite_quench_all"]

    for key, matrix in blocks.items():
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid finite-quench feature block {key}: {matrix.shape}")
        if matrix.shape[1] != len(names[key]):
            raise RuntimeError(f"feature names do not match {key}")
    return blocks, names


def finite_quench_feature_families(
    classical: np.ndarray,
    classical_names: tuple[str, ...],
    blocks: dict[str, np.ndarray],
    names: dict[str, tuple[str, ...]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Return the fixed finite-quench warning heads."""

    matrices: dict[str, np.ndarray] = {
        "final_static_qrc": blocks["final_static"],
        "quench_modes_qrc": blocks["quench_modes"],
        "reporter_quench_qrc": blocks["reporter_quench"],
        "finite_quench_qrc": blocks["finite_quench_all"],
        "all_qrc": blocks["qrc_all"],
    }
    output_names: dict[str, tuple[str, ...]] = {
        "final_static_qrc": names["final_static"],
        "quench_modes_qrc": names["quench_modes"],
        "reporter_quench_qrc": names["reporter_quench"],
        "finite_quench_qrc": names["finite_quench_all"],
        "all_qrc": names["qrc_all"],
    }
    label_map = {
        "final_static": "classical_plus_final_static",
        "quench_modes": "classical_plus_quench_modes",
        "reporter_quench": "classical_plus_reporter_quench",
        "finite_quench_all": "classical_plus_finite_quench",
        "qrc_all": "classical_plus_all",
    }
    for key, label in label_map.items():
        matrices[label] = np.concatenate([classical, blocks[key]], axis=1)
        output_names[label] = classical_names + names[key]
    return matrices, output_names
