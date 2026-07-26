from __future__ import annotations

import numpy as np
import pytest

from transition_forecasting.qrc.palindrome_noise_assay import (
    PalindromeNoiseAssayConfig,
    noise_scenarios,
)
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    _resolve_schedule,
    evolve_palindrome_probabilities,
)
from transition_forecasting.qrc.palindrome_rydberg_noise import (
    _apply_global_rxy_density,
    build_noisy_palindrome_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)
from transition_forecasting.qrc.temporal_rydberg_noise import TemporalNoiseSpec


def _reservoir() -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        omega_mod_fraction=0.60,
        step_duration_us=0.02,
        probe_fractions=(0.5, 1.0),
        max_phase_per_substep=0.5,
        shots=None,
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


def test_global_rxy_density_preserves_trace_and_hermiticity() -> None:
    density = np.zeros((2, 4, 4), dtype=complex)
    density[:, 0, 0] = 1.0
    output = _apply_global_rxy_density(
        density,
        np.asarray([0.2, 0.4]),
        np.asarray([0.0, 0.7]),
        n_atoms=2,
    )
    np.testing.assert_allclose(np.trace(output, axis1=1, axis2=2), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        output,
        output.conj().transpose(0, 2, 1),
        atol=1e-12,
    )


def test_ideal_density_palindrome_matches_statevector_probabilities() -> None:
    rng = np.random.default_rng(31)
    windows = rng.uniform(-0.5, 0.5, size=(1, 4, 2))
    reservoir = _reservoir()
    geometry = _geometry()
    schedule = _resolve_schedule("crossover_Ahalf_B_Ahalf")
    statevector, _ = evolve_palindrome_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=1.25,
        interactions=True,
        drive_phase_rad=0.0,
    )
    density, metadata = build_noisy_palindrome_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        TemporalNoiseSpec(name="ideal_density"),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
        interactions=True,
    )
    np.testing.assert_allclose(density, statevector, atol=1e-10, rtol=1e-10)
    assert metadata["max_trace_drift_before_renormalization"] < 1e-10
    assert metadata["max_hermiticity_error"] < 1e-12


def test_noisy_palindrome_probabilities_remain_normalized() -> None:
    windows = np.zeros((1, 3, 2), dtype=float)
    probabilities, metadata = build_noisy_palindrome_probabilities(
        windows,
        _reservoir(),
        _geometry(),
        _resolve_schedule("crossover_Ahalf_B_Ahalf"),
        TemporalNoiseSpec(
            name="combined",
            amplitude_damping_t1_us=100.0,
            dephasing_t2_us=50.0,
            depolarizing_probability=0.01,
        ),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    assert probabilities.shape == (1, 2, 64)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=1e-10)
    assert np.all(probabilities >= 0.0)
    assert metadata["total_substeps"] >= 3


def test_scenario_filter_always_keeps_ideal_reference() -> None:
    config = PalindromeNoiseAssayConfig(
        selected_scenarios=("depolarizing_p_0.01",),
    )
    scenarios = noise_scenarios(config)
    assert [scenario.name for scenario in scenarios] == [
        "ideal_density",
        "depolarizing_p_0.01",
    ]
    with pytest.raises(ValueError, match="unknown selected"):
        noise_scenarios(
            PalindromeNoiseAssayConfig(selected_scenarios=("not_a_scenario",))
        )


def test_submission_config_rejects_architecture_drift() -> None:
    PalindromeNoiseAssayConfig().validate()
    with pytest.raises(ValueError, match="level_instability"):
        PalindromeNoiseAssayConfig(representation="level_only").validate()
    with pytest.raises(ValueError, match="six_mode_density_curvature"):
        PalindromeNoiseAssayConfig(feature_bank="occupation_pair_raw").validate()
