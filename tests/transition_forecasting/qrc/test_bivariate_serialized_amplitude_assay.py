from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_capacity_assay import BivariateCapacityConfig
from transition_forecasting.qrc.bivariate_serialized_amplitude_assay import (
    BivariateSerializedAmplitudeConfig,
    _amplitude_drive,
    _metric_payload,
    build_serialized_capacity_targets,
    build_serialized_representations,
    evolve_amplitude_program_probabilities,
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


def test_configuration_preserves_exchange_pairs() -> None:
    config = BivariateSerializedAmplitudeConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        permutations=4,
    )
    config.validate()
    train_end = int(np.floor(config.samples * 0.60))
    validation_end = train_end + int(np.floor(config.samples * 0.20))
    assert train_end % 2 == 0
    assert validation_end % 2 == 0


def test_targets_separate_pointwise_within_and_cross_channel_terms() -> None:
    rng = np.random.default_rng(91)
    windows = rng.uniform(-1.0, 1.0, size=(160, 20, 2))
    config = BivariateCapacityConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seed=91,
    )
    targets = build_serialized_capacity_targets(windows, config)
    assert targets.values.shape == (160, 26)
    assert set(targets.metadata["group"]) == {
        "memory",
        "pointwise",
        "within_channel",
        "cross_channel",
        "order",
    }
    assert len(targets.metadata.loc[targets.metadata["group"].eq("within_channel")]) == 4
    assert len(targets.metadata.loc[targets.metadata["group"].eq("cross_channel")]) == 5
    pointwise = targets.values[:, targets.metadata["group"].eq("pointwise")]
    assert np.isfinite(pointwise).all()


def test_amplitude_drive_uses_one_feature_and_fixed_detuning() -> None:
    windows = np.zeros((4, 8, 2), dtype=float)
    windows[:, 3, 0] = np.array([-1.0, -0.5, 0.5, 1.0])
    windows[:, 3, 1] = 0.25
    omega, delta, phase = _amplitude_drive(
        windows,
        3,
        0,
        _reservoir(),
        0.0,
    )
    assert np.all(omega > 0)
    np.testing.assert_allclose(delta, 6.0)
    np.testing.assert_allclose(phase, 0.0)
    expected = 6.0 * (1.0 + 0.60 * windows[:, 3, 0])
    np.testing.assert_allclose(omega, expected)


def test_two_identical_half_substeps_reproduce_one_full_step() -> None:
    rng = np.random.default_rng(20260724)
    windows = rng.uniform(-1.0, 1.0, size=(4, 8, 2))
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    full, _ = evolve_amplitude_program_probabilities(
        windows,
        _reservoir(),
        geometry,
        (0,),
        program_name="full_u1",
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    repeated, metadata = evolve_amplitude_program_probabilities(
        windows,
        _reservoir(),
        geometry,
        (0, 0),
        program_name="repeated_u1",
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    np.testing.assert_allclose(repeated, full, atol=1e-12, rtol=1e-12)
    assert metadata["segment_duration_us"] == pytest.approx(0.0075)


def test_serialized_forward_reverse_are_exact_channel_mirrors() -> None:
    rng = np.random.default_rng(20260725)
    windows = rng.uniform(-1.0, 1.0, size=(4, 8, 2))
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    forward, _ = evolve_amplitude_program_probabilities(
        windows,
        _reservoir(),
        geometry,
        (0, 1),
        program_name="forward",
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    reverse_mirror, _ = evolve_amplitude_program_probabilities(
        windows[:, :, ::-1],
        _reservoir(),
        geometry,
        (1, 0),
        program_name="reverse",
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    np.testing.assert_allclose(reverse_mirror, forward, atol=1e-12, rtol=1e-12)


def test_representations_have_expected_equal_width_controls() -> None:
    rng = np.random.default_rng(7)
    raw = {
        "single_u1": rng.normal(size=(10, 63)),
        "single_u2": rng.normal(size=(10, 63)),
        "serialized_u1_u2": rng.normal(size=(10, 63)),
        "serialized_u2_u1": rng.normal(size=(10, 63)),
    }
    representations = build_serialized_representations(raw)
    assert representations["single_u1"].shape == (10, 63)
    assert representations["serialized_forward"].shape == (10, 63)
    assert representations["single_channels_concat"].shape == (10, 126)
    assert representations["serialized_mirrors_concat"].shape == (10, 126)


def test_metric_payload_keeps_within_and_cross_temporal_capacity_separate() -> None:
    rows = []
    for channel in (1, 2):
        for delay, capacity in ((1, 0.4), (2, 0.3), (5, 0.2)):
            rows.append(
                {
                    "task": f"memory_u{channel}_d{delay}",
                    "group": "memory",
                    "channel": channel,
                    "delay_a": delay,
                    "capacity": capacity,
                }
            )
        rows.append(
            {
                "task": f"pointwise_p2_u{channel}_d1",
                "group": "pointwise",
                "channel": channel,
                "delay_a": 1,
                "capacity": 0.05,
            }
        )
        rows.append(
            {
                "task": f"within_u{channel}_d1_d2",
                "group": "within_channel",
                "channel": channel,
                "delay_a": 1,
                "capacity": 0.07,
            }
        )
    rows.extend(
        [
            {
                "task": "cross_same_d1",
                "group": "cross_channel",
                "channel": np.nan,
                "delay_a": 1,
                "capacity": 0.11,
            },
            {
                "task": "cross_u1d1_u2d2",
                "group": "cross_channel",
                "channel": np.nan,
                "delay_a": 1,
                "capacity": 0.13,
            },
            {
                "task": "antisymmetric_order_d1_d2",
                "group": "order",
                "channel": np.nan,
                "delay_a": 1,
                "capacity": 0.03,
            },
        ]
    )
    payload = _metric_payload(pd.DataFrame(rows))
    assert payload["within_u1"] == pytest.approx(0.07)
    assert payload["within_u2"] == pytest.approx(0.07)
    assert payload["cross_same"] == pytest.approx(0.11)
    assert payload["cross_delayed"] == pytest.approx(0.13)
    assert payload["nonlinear_total"] == pytest.approx(0.48)
