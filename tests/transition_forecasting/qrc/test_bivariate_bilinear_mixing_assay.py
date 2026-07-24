from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.bivariate_bilinear_mixing_assay import (
    BILINEAR_SCHEDULES,
    BivariateBilinearMixingConfig,
    build_bilinear_representations,
    evolve_bilinear_schedule_probabilities,
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
        step_duration_us=0.02,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )


def test_bilinear_schedule_channel_mirrors_are_exact() -> None:
    rng = np.random.default_rng(20260724)
    windows = rng.uniform(-1.0, 1.0, size=(4, 8, 2))
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)

    left, _ = evolve_bilinear_schedule_probabilities(
        windows,
        _reservoir(),
        geometry,
        BILINEAR_SCHEDULES["d1_then_x2"],
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    mirrored, _ = evolve_bilinear_schedule_probabilities(
        windows[:, :, ::-1],
        _reservoir(),
        geometry,
        BILINEAR_SCHEDULES["d2_then_x1"],
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )

    np.testing.assert_allclose(mirrored, left, atol=1e-12, rtol=1e-12)


def test_reversing_longitudinal_transverse_order_changes_probabilities() -> None:
    rng = np.random.default_rng(20260725)
    windows = rng.uniform(-1.0, 1.0, size=(3, 8, 2))
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)

    forward, _ = evolve_bilinear_schedule_probabilities(
        windows,
        _reservoir(),
        geometry,
        BILINEAR_SCHEDULES["d1_then_x2"],
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    reverse, _ = evolve_bilinear_schedule_probabilities(
        windows,
        _reservoir(),
        geometry,
        BILINEAR_SCHEDULES["x2_then_d1"],
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )

    assert float(np.max(np.abs(forward - reverse))) > 1e-8


def test_interaction_off_bilinear_probabilities_are_normalized() -> None:
    rng = np.random.default_rng(20260726)
    windows = rng.uniform(-1.0, 1.0, size=(3, 8, 2))
    probabilities, metadata = evolve_bilinear_schedule_probabilities(
        windows,
        _reservoir(),
        StaggeredLadderGeometryConfig(row_spacing_um=9.0),
        BILINEAR_SCHEDULES["d1_then_x2"],
        interaction_scale=0.0,
        drive_phase_rad=0.0,
    )

    assert probabilities.shape == (3, 2, 64)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=1e-12)
    assert metadata["interactions_enabled"] is False


def test_bilinear_representations_have_expected_widths_and_contrast() -> None:
    rng = np.random.default_rng(88)
    raw = {
        "palindrome": rng.normal(size=(10, 63)),
        "d1_then_x2": rng.normal(size=(10, 63)),
        "d2_then_x1": rng.normal(size=(10, 63)),
        "x2_then_d1": rng.normal(size=(10, 63)),
        "x1_then_d2": rng.normal(size=(10, 63)),
    }
    representations = build_bilinear_representations(raw)

    assert representations["palindrome_control"].shape == (10, 63)
    assert representations["forward_mirror_concat"].shape == (10, 126)
    assert representations["reverse_mirror_concat"].shape == (10, 126)
    assert representations["commutator_contrast_concat"].shape == (10, 126)
    expected = np.concatenate(
        [
            raw["d1_then_x2"] - raw["x2_then_d1"],
            raw["d2_then_x1"] - raw["x1_then_d2"],
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        representations["commutator_contrast_concat"], expected
    )


def test_smoke_configuration_is_pair_preserving() -> None:
    config = BivariateBilinearMixingConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        permutations=4,
    )
    config.validate()
