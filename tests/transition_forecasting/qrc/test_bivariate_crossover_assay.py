from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    BivariateCrossoverConfig,
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def _schedule(name: str):
    return next(schedule for schedule in CROSSOVER_SCHEDULES if schedule.name == name)


def test_every_crossover_schedule_has_equal_branch_exposure() -> None:
    for schedule in CROSSOVER_SCHEDULES:
        schedule.validate()
        exposure = {
            branch: sum(
                fraction
                for local_branch, fraction in schedule.segments
                if local_branch == branch
            )
            for branch in ("A", "B")
        }
        np.testing.assert_allclose(exposure["A"], 0.5, atol=1e-15)
        np.testing.assert_allclose(exposure["B"], 0.5, atol=1e-15)


def test_mirrored_two_segment_schedules_are_exact_under_stream_swap() -> None:
    rng = np.random.default_rng(20260724)
    windows = rng.uniform(-1.0, 1.0, size=(4, 8, 2))
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        step_duration_us=0.03,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)

    original, _ = evolve_crossover_probabilities(
        windows,
        reservoir,
        geometry,
        _schedule("crossover_A_B"),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    mirrored, _ = evolve_crossover_probabilities(
        windows[:, :, ::-1],
        reservoir,
        geometry,
        _schedule("crossover_B_A"),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )

    np.testing.assert_allclose(mirrored, original, atol=1e-12, rtol=1e-12)


def test_mirrored_strang_schedules_are_exact_under_stream_swap() -> None:
    rng = np.random.default_rng(20260725)
    windows = rng.uniform(-1.0, 1.0, size=(3, 8, 2))
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        step_duration_us=0.03,
        probe_fractions=(1.0,),
        shots=None,
    )
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)

    original, _ = evolve_crossover_probabilities(
        windows,
        reservoir,
        geometry,
        _schedule("crossover_Ahalf_B_Ahalf"),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    mirrored, _ = evolve_crossover_probabilities(
        windows[:, :, ::-1],
        reservoir,
        geometry,
        _schedule("crossover_Bhalf_A_Bhalf"),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )

    np.testing.assert_allclose(mirrored, original, atol=1e-12, rtol=1e-12)


def test_crossover_feature_banks_have_expected_three_probe_widths() -> None:
    probabilities = np.full((5, 3, 64), 1.0 / 64.0)
    banks = build_crossover_feature_banks(probabilities)

    assert banks["six_mode_density_curvature"].shape == (5, 6)
    assert banks["occupation_pair_raw"].shape == (5, 63)
    assert banks["full_one_two_body"].shape == (5, 108)


def test_raw_bank_is_occupation_and_pair_prefix_of_each_probe_block() -> None:
    rng = np.random.default_rng(88)
    probabilities = rng.uniform(size=(4, 3, 64))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    banks = build_crossover_feature_banks(probabilities)
    full = banks["full_one_two_body"].reshape(4, 3, 36)
    expected = full[:, :, :21].reshape(4, 63)

    np.testing.assert_allclose(banks["occupation_pair_raw"], expected)


def test_smoke_configuration_is_valid() -> None:
    config = BivariateCrossoverConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        permutations=4,
    )
    config.validate()
