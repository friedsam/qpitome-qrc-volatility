from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.bivariate_capacity_dynamics import (
    evolve_sequential_probe_probabilities,
)
from transition_forecasting.qrc.bivariate_encoding_symmetry_assay import (
    ENCODING_CASES,
    _channel_memory_rows,
    evolve_encoding_case_probabilities,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    evolve_ladder_probe_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def _reservoir() -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.01,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )


def _windows() -> np.ndarray:
    rng = np.random.default_rng(20260724)
    return rng.uniform(-1.0, 1.0, size=(5, 8, 2))


def _case(name: str):
    return next(case for case in ENCODING_CASES if case.name == name)


def test_encoding_cases_form_complete_route_slot_factorial() -> None:
    assert len(ENCODING_CASES) == 6
    sequential = [case for case in ENCODING_CASES if not case.simultaneous]
    combinations = {
        (case.detuning_channel, case.amplitude_channel, case.slot_order)
        for case in sequential
    }
    assert combinations == {
        (0, 1, ("detuning", "amplitude")),
        (1, 0, ("detuning", "amplitude")),
        (0, 1, ("amplitude", "detuning")),
        (1, 0, ("amplitude", "detuning")),
    }
    for case in ENCODING_CASES:
        assert {case.route_for_channel(0), case.route_for_channel(1)} == {
            "detuning",
            "amplitude",
        }


def test_simultaneous_cases_match_incumbent_with_paired_channel_order() -> None:
    windows = _windows()
    reservoir = _reservoir()
    geometry = StaggeredLadderGeometryConfig()

    original, _ = evolve_encoding_case_probabilities(
        windows,
        reservoir,
        geometry,
        _case("simultaneous_original"),
        interaction_scale=1.25,
    )
    expected_original, _ = evolve_ladder_probe_probabilities(
        windows,
        reservoir,
        geometry,
        interaction_scale=1.25,
        condition="ordered",
    )
    np.testing.assert_allclose(original, expected_original, atol=1e-12, rtol=1e-12)

    swapped, _ = evolve_encoding_case_probabilities(
        windows,
        reservoir,
        geometry,
        _case("simultaneous_swapped"),
        interaction_scale=1.25,
    )
    expected_swapped, _ = evolve_ladder_probe_probabilities(
        windows[:, :, ::-1],
        reservoir,
        geometry,
        interaction_scale=1.25,
        condition="ordered",
    )
    np.testing.assert_allclose(swapped, expected_swapped, atol=1e-12, rtol=1e-12)


def test_original_sequential_case_matches_existing_noncommuting_encoder() -> None:
    windows = _windows()
    reservoir = _reservoir()
    geometry = StaggeredLadderGeometryConfig()

    observed, metadata = evolve_encoding_case_probabilities(
        windows,
        reservoir,
        geometry,
        _case("sequential_det_first_original"),
        interaction_scale=1.25,
    )
    expected, _ = evolve_sequential_probe_probabilities(
        windows,
        reservoir,
        geometry,
        interaction_scale=1.25,
        first_slot_fraction=0.5,
        second_phase_rad=float(np.pi / 2.0),
    )

    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)
    assert metadata["slot_order"] == ["detuning", "amplitude"]
    assert np.allclose(observed.sum(axis=2), 1.0)


def test_channel_summary_tracks_route_and_slot_without_relabeling_targets() -> None:
    case = _case("sequential_amp_first_swapped")
    rows = []
    for channel in (1, 2):
        for delay, capacity in ((1, 0.5), (2, 0.4), (5, 0.3), (8, 0.1)):
            rows.append(
                {
                    "group": "memory",
                    "channel": channel,
                    "delay_a": delay,
                    "capacity": capacity + 0.01 * channel,
                }
            )
    summary = _channel_memory_rows(
        pd.DataFrame(rows),
        case,
        seed=7,
        feature_bank="full_one_two_body",
    )

    channel_one = next(row for row in summary if row["channel"] == 1)
    channel_two = next(row for row in summary if row["channel"] == 2)
    assert channel_one["control_route"] == "amplitude"
    assert channel_one["slot"] == "first"
    assert channel_two["control_route"] == "detuning"
    assert channel_two["slot"] == "second"
