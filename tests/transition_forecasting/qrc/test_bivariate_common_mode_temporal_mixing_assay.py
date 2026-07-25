from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_capacity_assay import BivariateCapacityConfig
from transition_forecasting.qrc.bivariate_common_mode_temporal_mixing_assay import (
    BivariateCommonModeTemporalMixingConfig,
    PROGRAMS,
    PRODUCT_FIELDS,
    _common_scalar,
    _metric_payload,
    _segment_payloads,
    build_common_mode_capacity_targets,
    build_common_mode_representations,
    evolve_common_mode_program_probabilities,
    generate_common_mode_windows,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
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
        step_duration_us=0.015,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )


def _config() -> BivariateCommonModeTemporalMixingConfig:
    return BivariateCommonModeTemporalMixingConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        permutations=4,
    )


def test_common_mode_windows_duplicate_scalar_stream_exactly() -> None:
    windows = generate_common_mode_windows(_config(), seed=20260724)
    assert windows.shape == (160, 20, 2)
    np.testing.assert_array_equal(windows[:, :, 0], windows[:, :, 1])
    np.testing.assert_array_equal(_common_scalar(windows), windows[:, :, 0])


def test_common_mode_rejects_nonidentical_channels() -> None:
    windows = generate_common_mode_windows(_config(), seed=20260724)
    windows[0, 0, 1] += 1e-9
    with pytest.raises(ValueError, match="exactly duplicated"):
        _common_scalar(windows)


def test_targets_are_univariate_and_separate_pointwise_from_temporal_products() -> None:
    windows = generate_common_mode_windows(_config(), seed=91)
    capacity = BivariateCapacityConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seed=91,
    )
    targets = build_common_mode_capacity_targets(windows, capacity)
    assert targets.values.shape == (160, 13)
    assert set(targets.metadata["group"]) == {
        "memory",
        "pointwise",
        "temporal_product",
    }
    assert len(
        targets.metadata.loc[targets.metadata["group"].eq("temporal_product")]
    ) == 5
    assert set(
        targets.metadata.loc[
            targets.metadata["group"].eq("temporal_product"), "task"
        ]
    ) == set(PRODUCT_FIELDS)


def test_single_route_programs_modulate_only_one_input_coefficient() -> None:
    windows = generate_common_mode_windows(_config(), seed=8)[:4, :8]
    reservoir = _reservoir()
    amplitude = _segment_payloads(windows, 3, "amplitude_only", reservoir, 0.0)
    detuning = _segment_payloads(windows, 3, "detuning_only", reservoir, 0.0)
    assert len(amplitude) == len(detuning) == 1
    omega_a, delta_a, _, duration_a = amplitude[0]
    omega_d, delta_d, _, duration_d = detuning[0]
    assert duration_a == pytest.approx(reservoir.step_duration_us)
    assert duration_d == pytest.approx(reservoir.step_duration_us)
    np.testing.assert_allclose(delta_a, reservoir.delta_center_rad_us)
    np.testing.assert_allclose(omega_d, reservoir.omega_base_rad_us)
    assert np.std(omega_a) > 0
    assert np.std(delta_d) > 0


def test_area_matched_simultaneous_integrals_equal_ordered_dx() -> None:
    windows = generate_common_mode_windows(_config(), seed=9)[:5, :8]
    reservoir = _reservoir()
    simultaneous = _segment_payloads(
        windows, 2, "simultaneous_area_matched", reservoir, 0.0
    )
    ordered = _segment_payloads(windows, 2, "ordered_dx", reservoir, 0.0)
    omega_sim, delta_sim, _, duration_sim = simultaneous[0]
    omega_order = sum(omega * duration for omega, _, _, duration in ordered)
    delta_order = sum(delta * duration for _, delta, _, duration in ordered)
    np.testing.assert_allclose(omega_sim * duration_sim, omega_order)
    np.testing.assert_allclose(delta_sim * duration_sim, delta_order)


def test_switching_programs_preserve_total_timestep_duration() -> None:
    windows = generate_common_mode_windows(_config(), seed=10)[:3, :8]
    reservoir = _reservoir()
    expected_counts = {
        "ordered_dx": 2,
        "ordered_xd": 2,
        "palindrome_dxd": 3,
        "palindrome_xdx": 3,
    }
    for program, count in expected_counts.items():
        segments = _segment_payloads(windows, 1, program, reservoir, 0.0)
        assert len(segments) == count
        assert sum(segment[3] for segment in segments) == pytest.approx(
            reservoir.step_duration_us
        )


def test_representations_supply_equal_width_switching_controls() -> None:
    rng = np.random.default_rng(7)
    raw = {name: rng.normal(size=(10, 63)) for name in PROGRAMS}
    representations = build_common_mode_representations(raw)
    for program in PROGRAMS:
        assert representations[program].shape == (10, 63)
    assert representations["single_routes_concat"].shape == (10, 126)
    assert representations["ordered_mirrors_concat"].shape == (10, 126)
    assert representations["palindrome_mirrors_concat"].shape == (10, 126)


def test_metric_payload_keeps_temporal_products_separate() -> None:
    rows = [
        {
            "task": f"memory_d{delay}",
            "group": "memory",
            "delay_a": delay,
            "capacity": capacity,
        }
        for delay, capacity in ((1, 0.4), (2, 0.3), (5, 0.2), (8, 0.1))
    ]
    rows.extend(
        [
            {
                "task": "pointwise_p2_d1",
                "group": "pointwise",
                "delay_a": 1,
                "capacity": 0.05,
            },
            {
                "task": "pointwise_p2_d5",
                "group": "pointwise",
                "delay_a": 5,
                "capacity": 0.04,
            },
        ]
    )
    for index, field in enumerate(PRODUCT_FIELDS, start=1):
        rows.append(
            {
                "task": field,
                "group": "temporal_product",
                "delay_a": index,
                "capacity": 0.01 * index,
            }
        )
    payload = _metric_payload(pd.DataFrame(rows))
    assert payload["delay5"] == pytest.approx(0.2)
    assert payload["pointwise_total"] == pytest.approx(0.09)
    assert payload["temporal_total"] == pytest.approx(0.15)
    assert payload[PRODUCT_FIELDS[-1]] == pytest.approx(0.05)


def test_ordered_programs_produce_normalized_distinct_states() -> None:
    windows = generate_common_mode_windows(_config(), seed=20260725)[:4, :8]
    reservoir = _reservoir()
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    forward, _ = evolve_common_mode_program_probabilities(
        windows,
        reservoir,
        geometry,
        "ordered_dx",
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    reverse, _ = evolve_common_mode_program_probabilities(
        windows,
        reservoir,
        geometry,
        "ordered_xd",
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    np.testing.assert_allclose(forward.sum(axis=2), 1.0)
    np.testing.assert_allclose(reverse.sum(axis=2), 1.0)
    assert not np.allclose(forward, reverse)
