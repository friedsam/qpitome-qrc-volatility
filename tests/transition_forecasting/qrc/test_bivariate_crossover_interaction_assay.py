from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.bivariate_crossover_interaction_assay import (
    BivariateCrossoverInteractionConfig,
    _evolve_interaction_off_probabilities,
    equivalent_geometry_factor,
    minimum_pair_distance_um,
)
from transition_forecasting.qrc.bivariate_crossover_stability_assay import (
    generate_exchange_symmetric_windows,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def _smoke_config() -> BivariateCrossoverInteractionConfig:
    return BivariateCrossoverInteractionConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        interaction_scales=(0.0, 0.5, 1.25, 2.0),
        permutations=4,
    )


def test_default_interaction_grid_contains_off_and_incumbent() -> None:
    config = BivariateCrossoverInteractionConfig()
    config.validate()

    assert 0.0 in config.interaction_scales
    assert 1.25 in config.interaction_scales
    assert config.step_duration_us == 0.02


def test_equivalent_geometry_factor_obeys_inverse_sixth_power() -> None:
    for scale in (0.25, 0.5, 1.0, 1.25, 2.0, 3.0):
        factor = equivalent_geometry_factor(scale)
        np.testing.assert_allclose(factor**-6, scale, rtol=1e-12, atol=1e-12)

    assert np.isnan(equivalent_geometry_factor(0.0))


def test_current_geometry_has_expected_minimum_spacing() -> None:
    geometry = StaggeredLadderGeometryConfig(
        longitudinal_spacing_um=8.5,
        row_spacing_um=9.0,
    )
    np.testing.assert_allclose(minimum_pair_distance_um(geometry), 8.5, atol=1e-12)


def test_interaction_off_palindrome_returns_normalized_probabilities() -> None:
    config = _smoke_config()
    stability = config.stability_config()
    windows = generate_exchange_symmetric_windows(stability, seed=20260724)[:4]
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        step_duration_us=0.02,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)

    probabilities, metadata = _evolve_interaction_off_probabilities(
        windows,
        reservoir,
        geometry,
        drive_phase_rad=0.0,
    )

    assert probabilities.shape == (4, 2, 64)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=1e-12)
    assert metadata["interaction_scale"] == 0.0
    assert metadata["interactions_enabled"] is False


def test_smoke_configuration_preserves_exchange_pairs() -> None:
    config = _smoke_config()
    config.validate()
    stability = config.stability_config()
    windows = generate_exchange_symmetric_windows(stability, seed=20260724)

    np.testing.assert_allclose(windows[1::2], windows[0::2, :, ::-1], atol=0.0, rtol=0.0)
