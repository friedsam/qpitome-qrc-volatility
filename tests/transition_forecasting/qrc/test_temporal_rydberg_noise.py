from __future__ import annotations

import numpy as np
import pytest

from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    build_temporal_rydberg_chain_features,
)
from transition_forecasting.qrc.temporal_rydberg_noise import (
    TemporalNoiseSpec,
    _amplitude_damping_kraus,
    _apply_local_kraus_channel,
    _depolarizing_kraus,
    build_noisy_temporal_rydberg_features,
)
from transition_forecasting.qrc.temporal_rydberg_noise_assay import (
    TemporalNoiseAssayConfig,
    noise_scenarios,
)


def test_noise_spec_validation() -> None:
    TemporalNoiseSpec(name="ideal").validate()
    with pytest.raises(ValueError):
        TemporalNoiseSpec(name="bad", amplitude_damping_t1_us=0.0).validate()
    with pytest.raises(ValueError):
        TemporalNoiseSpec(name="bad", depolarizing_probability=1.0).validate()


def test_amplitude_damping_is_trace_preserving_and_reduces_excitation() -> None:
    density = np.zeros((1, 2, 2), dtype=complex)
    density[0, 1, 1] = 1.0
    output = _apply_local_kraus_channel(
        density,
        _amplitude_damping_kraus(dt_us=1.0, t1_us=10.0),
        n_atoms=1,
    )
    assert np.trace(output[0]).real == pytest.approx(1.0, abs=1e-12)
    assert output[0, 1, 1].real < 1.0
    assert output[0, 0, 0].real > 0.0


def test_depolarizing_channel_moves_ground_state_toward_mixed_state() -> None:
    density = np.zeros((1, 2, 2), dtype=complex)
    density[0, 0, 0] = 1.0
    output = _apply_local_kraus_channel(
        density,
        _depolarizing_kraus(
            dt_us=1.0,
            total_time_us=1.0,
            total_probability=0.2,
        ),
        n_atoms=1,
    )
    assert np.trace(output[0]).real == pytest.approx(1.0, abs=1e-12)
    assert output[0, 0, 0].real == pytest.approx(0.9, abs=1e-12)
    assert output[0, 1, 1].real == pytest.approx(0.1, abs=1e-12)


def test_ideal_density_engine_matches_statevector_features() -> None:
    rng = np.random.default_rng(7)
    windows = rng.uniform(-0.5, 0.5, size=(2, 4, 2))
    config = TemporalRydbergChainConfig(
        n_atoms=3,
        spacing_short_um=8.5,
        spacing_long_um=10.0,
        defect_edge=1,
        defect_offset_um=0.2,
        step_duration_us=0.02,
        probe_fractions=(0.5, 1.0),
        max_phase_per_substep=0.5,
        max_substeps_per_step=64,
        shots=None,
    )
    statevector, _ = build_temporal_rydberg_chain_features(
        windows,
        config,
        condition="ordered",
    )
    density, metadata = build_noisy_temporal_rydberg_features(
        windows,
        config,
        TemporalNoiseSpec(name="ideal"),
    )
    np.testing.assert_allclose(density, statevector, atol=1e-10, rtol=1e-10)
    assert metadata["max_trace_drift_before_renormalization"] < 1e-10
    assert metadata["max_hermiticity_error"] < 1e-12


def test_default_assay_includes_required_and_physical_channels() -> None:
    scenarios = noise_scenarios(TemporalNoiseAssayConfig())
    assert any(value.amplitude_damping_t1_us is not None for value in scenarios)
    assert any(value.depolarizing_probability > 0.0 for value in scenarios)
    assert any(value.dephasing_t2_us is not None for value in scenarios)
