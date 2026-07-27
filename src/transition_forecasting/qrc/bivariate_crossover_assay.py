"""Canonical A/B/A schedule and feature bank for the Case151 assay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from transition_forecasting.qrc.bivariate_capacity_dynamics import build_feature_banks
from transition_forecasting.qrc.ladder_finite_shot_sampling import validate_probe_probabilities
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig

BranchName = Literal["A", "B"]
CrossoverFeatureBank = Literal[
    "six_mode_density_curvature", "occupation_pair_raw", "full_one_two_body"
]


@dataclass(frozen=True)
class CrossoverSchedule:
    name: str
    segments: tuple[tuple[BranchName, float], ...]

    def validate(self) -> None:
        if not self.name or not self.segments:
            raise ValueError("crossover schedule must be named and nonempty")
        if any(branch not in ("A", "B") for branch, _ in self.segments):
            raise ValueError("crossover branches must be A or B")
        if any(fraction <= 0 for _, fraction in self.segments):
            raise ValueError("crossover segment fractions must be positive")
        if not np.isclose(sum(fraction for _, fraction in self.segments), 1.0):
            raise ValueError("crossover segment fractions must sum to one")


CROSSOVER_SCHEDULES: tuple[CrossoverSchedule, ...] = (
    CrossoverSchedule("crossover_A_B", (("A", 0.5), ("B", 0.5))),
    CrossoverSchedule("crossover_B_A", (("B", 0.5), ("A", 0.5))),
    CrossoverSchedule(
        "crossover_Ahalf_B_Ahalf", (("A", 0.25), ("B", 0.5), ("A", 0.25))
    ),
    CrossoverSchedule(
        "crossover_Bhalf_A_Bhalf", (("B", 0.25), ("A", 0.5), ("B", 0.25))
    ),
)


def _branch_drive(
    windows: np.ndarray,
    step: int,
    branch: BranchName,
    reservoir: TemporalRydbergChainConfig,
    phase_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if branch == "A":
        detuning_channel, amplitude_channel = 0, 1
    elif branch == "B":
        detuning_channel, amplitude_channel = 1, 0
    else:
        raise ValueError(f"unsupported crossover branch: {branch}")
    detuning_input = windows[:, step, detuning_channel]
    amplitude_input = windows[:, step, amplitude_channel]
    omega = reservoir.omega_base_rad_us * (
        1.0 + reservoir.omega_mod_fraction * amplitude_input
    )
    if np.any(omega <= 0):
        raise ValueError("crossover amplitude encoding became nonpositive")
    delta = reservoir.delta_center_rad_us + reservoir.delta_span_rad_us * detuning_input
    phase = np.full(len(windows), float(phase_rad), dtype=float)
    return omega, delta, phase


def build_crossover_feature_banks(
    probabilities: np.ndarray,
) -> dict[CrossoverFeatureBank, np.ndarray]:
    base = build_feature_banks(probabilities)
    full = np.asarray(base["full_one_two_body"], dtype=float)
    probes = validate_probe_probabilities(probabilities).shape[1]
    raw_indices = np.asarray(
        [
            probe * 36 + offset
            for probe in range(probes)
            for offset in range(21)
        ],
        dtype=int,
    )
    banks: dict[CrossoverFeatureBank, np.ndarray] = {
        "six_mode_density_curvature": np.asarray(
            base["six_mode_density_curvature"], dtype=float
        ),
        "occupation_pair_raw": full[:, raw_indices],
        "full_one_two_body": full,
    }
    expected = {
        "six_mode_density_curvature": probes * 2,
        "occupation_pair_raw": probes * 21,
        "full_one_two_body": probes * 36,
    }
    for name, matrix in banks.items():
        if matrix.shape != (len(full), expected[name]):
            raise RuntimeError(f"unexpected feature width for {name}: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"non-finite crossover features in {name}")
    return banks
