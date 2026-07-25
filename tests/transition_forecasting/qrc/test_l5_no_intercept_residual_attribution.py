from __future__ import annotations

import numpy as np
import pytest

from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.l5_no_intercept_residual_attribution import (
    L5NoInterceptResidualAttributionConfig,
    _fit_no_intercept_readout,
    endpoint_preserving_shuffle,
    evolve_crossover_control_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import StaggeredLadderGeometryConfig


def _schedule(name: str):
    return next(item for item in CROSSOVER_SCHEDULES if item.name == name)


def _reservoir() -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.020,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
    )


def test_config_requires_partitioned_development_folds() -> None:
    with pytest.raises(ValueError, match="partition"):
        L5NoInterceptResidualAttributionConfig(
            folds=(4, 5),
            selection_folds=(4, 5),
            confirmation_folds=(5,),
        ).validate()


def test_endpoint_preserving_shuffle_changes_history_not_endpoint() -> None:
    values = np.arange(5 * 8 * 2, dtype=float).reshape(5, 8, 2)
    shuffled = endpoint_preserving_shuffle(values, seed=17)
    assert np.array_equal(shuffled[:, -1, :], values[:, -1, :])
    assert not np.array_equal(shuffled[:, :-1, :], values[:, :-1, :])
    for row in range(len(values)):
        assert sorted(shuffled[row, :-1, 0]) == sorted(values[row, :-1, 0])
        assert sorted(shuffled[row, :-1, 1]) == sorted(values[row, :-1, 1])


def test_interacting_control_path_matches_primary_crossover_exactly() -> None:
    rng = np.random.default_rng(23)
    windows = rng.uniform(-0.8, 0.8, size=(6, 8, 2))
    reservoir = _reservoir()
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    schedule = _schedule("crossover_Ahalf_B_Ahalf")
    primary, _ = evolve_crossover_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    control = evolve_crossover_control_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=1.25,
        interactions=True,
    )
    assert np.allclose(primary, control, atol=1e-13, rtol=1e-13)


def test_interaction_off_is_a_genuine_dynamics_control() -> None:
    rng = np.random.default_rng(29)
    windows = rng.uniform(-0.8, 0.8, size=(6, 8, 2))
    reservoir = _reservoir()
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    schedule = _schedule("crossover_Ahalf_B_Ahalf")
    interacting = evolve_crossover_control_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=1.25,
        interactions=True,
    )
    noninteracting = evolve_crossover_control_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=1.25,
        interactions=False,
    )
    assert not np.allclose(interacting, noninteracting, atol=1e-10, rtol=1e-10)


def test_no_intercept_readout_has_exact_feature_attribution() -> None:
    rng = np.random.default_rng(31)
    features = rng.normal(size=(80, 7))
    coefficients = rng.normal(size=(3, 7))
    residual = features @ coefficients.T + rng.normal(0.0, 0.01, size=(80, 3))
    fit = np.zeros(80, dtype=bool)
    fit[:60] = True
    correction, diagnostics, transformed, fitted = _fit_no_intercept_readout(
        features,
        residual,
        fit,
        alpha=0.1,
    )
    assert diagnostics["fit_intercept"] is False
    assert diagnostics["intercept_max_abs"] == 0.0
    assert diagnostics["zero_feature_max_abs_correction"] == 0.0
    assert diagnostics["manual_reconstruction_max_abs_error"] <= 1e-12
    assert np.array_equal(correction, transformed @ fitted.T)


def test_no_intercept_readout_cannot_learn_a_constant_residual() -> None:
    rng = np.random.default_rng(37)
    features = rng.normal(size=(100, 9))
    residual = np.full((100, 2), 0.75, dtype=float)
    fit = np.zeros(100, dtype=bool)
    fit[:80] = True
    correction, diagnostics, _, _ = _fit_no_intercept_readout(
        features,
        residual,
        fit,
        alpha=1.0,
    )
    assert diagnostics["intercept_max_abs"] == 0.0
    assert np.max(np.abs(correction[fit].mean(axis=0))) <= 1e-12
