from __future__ import annotations

import numpy as np
import pytest

from transition_forecasting.qrc.bivariate_crossover_assay import (
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.ladder_mode_readout_tools import ladder_mode_weights
from transition_forecasting.qrc.palindrome_scaling_assay import (
    PalindromeScalingConfig,
    evolve_scaling_palindrome_probabilities,
    generalized_staggered_ladder_positions,
    probabilities_to_scaling_features,
    resource_scaling_table,
    size_normalized_mode_weights,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    staggered_ladder_positions,
)


def _geometry() -> StaggeredLadderGeometryConfig:
    return StaggeredLadderGeometryConfig(
        longitudinal_spacing_um=8.5,
        row_spacing_um=9.0,
        stagger_fraction=0.35,
        bottom_spacing_scale=1.05,
        defect_site=4,
        defect_dx_um=0.35,
        defect_dy_um=-0.40,
    )


def _reservoir(n_atoms: int) -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=n_atoms,
        defect_edge=min(1, n_atoms - 2),
        omega_mod_fraction=0.60,
        step_duration_us=0.02,
        probe_fractions=(0.25, 0.5, 1.0),
        max_phase_per_substep=1.0,
        max_substeps_per_step=64,
        shots=None,
    )


def test_generalized_geometry_preserves_exact_six_atom_ladder() -> None:
    geometry = _geometry()
    np.testing.assert_allclose(
        generalized_staggered_ladder_positions(6, geometry),
        staggered_ladder_positions(geometry),
        atol=0.0,
        rtol=0.0,
    )
    for n_atoms in (5, 7, 12, 20):
        positions = generalized_staggered_ladder_positions(n_atoms, geometry)
        assert positions.shape == (n_atoms, 2)
        assert np.isfinite(positions).all()
        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )
        np.fill_diagonal(distances, np.inf)
        assert distances.min() > 0.0


def test_six_atom_mode_weights_match_incumbent_bank() -> None:
    expected = ladder_mode_weights()[:, [0, 2]]
    observed = size_normalized_mode_weights(6)
    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
    for n_atoms in (5, 7, 12, 20):
        weights = size_normalized_mode_weights(n_atoms)
        assert weights.shape == (n_atoms, 2)
        np.testing.assert_allclose(weights.T @ weights, np.eye(2), atol=1e-12)


def test_six_atom_features_match_existing_density_curvature_bank() -> None:
    rng = np.random.default_rng(19)
    probabilities = rng.random((3, 3, 64))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    observed = probabilities_to_scaling_features(probabilities, 6)
    expected = build_crossover_feature_banks(probabilities)[
        "six_mode_density_curvature"
    ]
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_small_exact_scaling_evolution_has_fixed_feature_width() -> None:
    windows = np.zeros((2, 4, 2), dtype=float)
    probabilities, metadata = evolve_scaling_palindrome_probabilities(
        windows,
        _reservoir(5),
        _geometry(),
        schedule_name="crossover_Ahalf_B_Ahalf",
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    assert probabilities.shape == (2, 3, 32)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=1e-12)
    features = probabilities_to_scaling_features(probabilities, 5)
    assert features.shape == (2, 6)
    assert metadata["state_dimension"] == 32
    assert metadata["total_substeps"] >= 12


def test_resource_table_covers_five_through_twenty() -> None:
    frame = resource_scaling_table(
        5,
        20,
        samples=48,
        exact_atom_counts=(5, 6, 7),
    )
    assert frame["n_atoms"].tolist() == list(range(5, 21))
    row_20 = frame.loc[frame["n_atoms"].eq(20)].iloc[0]
    assert row_20["state_dimension"] == 2**20
    assert row_20["complex_state_bytes_per_sample"] == 16 * 2**20
    assert row_20["raw_batch_state_gib"] == pytest.approx(0.75)
    assert frame.loc[frame["n_atoms"].eq(6), "exact_simulated"].item()
    assert not frame.loc[frame["n_atoms"].eq(20), "exact_simulated"].item()


def test_scaling_config_rejects_contract_drift() -> None:
    PalindromeScalingConfig().validate()
    with pytest.raises(ValueError, match="level_instability"):
        PalindromeScalingConfig(representation="level_only").validate()
    with pytest.raises(ValueError, match="fixed density-curvature"):
        PalindromeScalingConfig(feature_bank="occupation_pair_raw").validate()
    with pytest.raises(ValueError, match="above resource range"):
        PalindromeScalingConfig(
            exact_atom_counts=(5, 21), resource_max_atoms=20
        ).validate()
