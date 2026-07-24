from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_bilinear_mixing_assay import (
    BILINEAR_SCHEDULES,
    evolve_bilinear_schedule_probabilities,
)
from transition_forecasting.qrc.bivariate_bilinear_slot_fraction_assay import (
    BivariateBilinearSlotFractionConfig,
    _mixing_decomposition,
    build_all_orders_joint,
    evolve_fractional_bilinear_probabilities,
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
        step_duration_us=0.015,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )


def test_smoke_configuration_is_pair_preserving_and_contains_baseline() -> None:
    config = BivariateBilinearSlotFractionConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        detuning_fractions=(0.4, 0.5, 0.7),
        permutations=4,
    )
    config.validate()
    train_end = int(np.floor(config.samples * 0.60))
    validation_end = train_end + int(np.floor(config.samples * 0.20))
    assert train_end % 2 == 0
    assert validation_end % 2 == 0
    assert 0.5 in config.detuning_fractions


def test_slot_fraction_configuration_rejects_endpoints_and_duplicates() -> None:
    common = {
        "samples": 160,
        "sequence_length": 20,
        "memory_delays": (1, 2, 4, 5, 8, 12),
        "seeds": (20260724,),
        "permutations": 4,
    }
    with pytest.raises(ValueError, match="detuning fractions"):
        BivariateBilinearSlotFractionConfig(
            **common, detuning_fractions=(0.0, 0.5)
        ).validate()
    with pytest.raises(ValueError, match="detuning fractions"):
        BivariateBilinearSlotFractionConfig(
            **common, detuning_fractions=(0.5, 1.0)
        ).validate()
    with pytest.raises(ValueError, match="detuning fractions"):
        BivariateBilinearSlotFractionConfig(
            **common, detuning_fractions=(0.5, 0.5)
        ).validate()


def test_half_fraction_reproduces_original_bilinear_evolution() -> None:
    rng = np.random.default_rng(20260724)
    windows = rng.uniform(-1.0, 1.0, size=(4, 8, 2))
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    routes = BILINEAR_SCHEDULES["d1_then_x2"]

    reference, _ = evolve_bilinear_schedule_probabilities(
        windows,
        _reservoir(),
        geometry,
        routes,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    fractional, metadata = evolve_fractional_bilinear_probabilities(
        windows,
        _reservoir(),
        geometry,
        routes,
        detuning_fraction=0.5,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )

    np.testing.assert_allclose(fractional, reference, atol=1e-12, rtol=1e-12)
    assert metadata["detuning_duration_us"] == pytest.approx(0.0075)
    assert metadata["amplitude_duration_us"] == pytest.approx(0.0075)


def test_unequal_fraction_preserves_exact_channel_mirror() -> None:
    rng = np.random.default_rng(20260725)
    windows = rng.uniform(-1.0, 1.0, size=(4, 8, 2))
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)

    left, _ = evolve_fractional_bilinear_probabilities(
        windows,
        _reservoir(),
        geometry,
        BILINEAR_SCHEDULES["d1_then_x2"],
        detuning_fraction=0.7,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    mirrored, _ = evolve_fractional_bilinear_probabilities(
        windows[:, :, ::-1],
        _reservoir(),
        geometry,
        BILINEAR_SCHEDULES["d2_then_x1"],
        detuning_fraction=0.7,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )

    np.testing.assert_allclose(mirrored, left, atol=1e-12, rtol=1e-12)


def test_all_orders_joint_has_expected_width() -> None:
    rng = np.random.default_rng(88)
    raw = {
        name: rng.normal(size=(10, 63))
        for name in BILINEAR_SCHEDULES
    }
    matrix = build_all_orders_joint(raw)
    assert matrix.shape == (10, 252)


def test_mixing_decomposition_separates_same_and_delayed_tasks() -> None:
    metrics = pd.DataFrame(
        {
            "task": [
                "mix_same_d1",
                "mix_u1d1_u2d2",
                "mix_u1d2_u2d1",
                "memory_u1_d1",
            ],
            "group": ["mixing", "mixing", "mixing", "memory"],
            "capacity": [0.2, 0.03, 0.04, 0.8],
        }
    )
    result = _mixing_decomposition(metrics)
    assert result["same_lag_mixing"] == pytest.approx(0.2)
    assert result["delayed_mixing"] == pytest.approx(0.07)
