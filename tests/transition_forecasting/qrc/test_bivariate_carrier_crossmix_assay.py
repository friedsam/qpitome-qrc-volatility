from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_bilinear_mixing_assay import (
    BILINEAR_SCHEDULES,
)
from transition_forecasting.qrc.bivariate_carrier_crossmix_assay import (
    CARRIER_MASKS,
    BivariateCarrierCrossmixConfig,
    _carrier_vs_pure,
    _mixing_decomposition,
    build_architecture_representations,
    carrier_drive,
    evolve_carrier_mask_probabilities,
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


def test_configuration_is_pair_preserving_and_fixed_at_near_pass() -> None:
    config = BivariateCarrierCrossmixConfig(
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
    assert config.step_duration_us == pytest.approx(0.015)
    assert config.interaction_scale == pytest.approx(1.25)


def test_carrier_masks_are_full_rank_and_channel_swapped_in_pairs() -> None:
    identity = np.asarray(CARRIER_MASKS["identity"], dtype=float)
    identity_swap = np.asarray(CARRIER_MASKS["identity_channel_swap"], dtype=float)
    hadamard = np.asarray(CARRIER_MASKS["hadamard"], dtype=float)
    hadamard_swap = np.asarray(CARRIER_MASKS["hadamard_channel_swap"], dtype=float)
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]])

    assert np.linalg.matrix_rank(identity) == 2
    assert np.linalg.matrix_rank(hadamard) == 2
    np.testing.assert_allclose(identity_swap, identity @ swap, atol=1e-15)
    np.testing.assert_allclose(hadamard_swap, hadamard @ swap, atol=1e-15)
    np.testing.assert_allclose(hadamard @ hadamard.T, np.eye(2), atol=1e-15)


def test_carrier_drive_keeps_amplitude_and_detuning_active() -> None:
    windows = np.zeros((5, 8, 2), dtype=float)
    omega, delta, phase = carrier_drive(
        windows,
        3,
        np.asarray(CARRIER_MASKS["hadamard"], dtype=float),
        _reservoir(),
        0.0,
    )
    np.testing.assert_allclose(omega, np.full(5, 6.0))
    np.testing.assert_allclose(delta, np.full(5, 6.0))
    np.testing.assert_allclose(phase, np.zeros(5))
    assert np.all(omega > 0)
    assert np.all(delta != 0)


@pytest.mark.parametrize(
    ("mask_name", "swapped_name"),
    [
        ("identity", "identity_channel_swap"),
        ("hadamard", "hadamard_channel_swap"),
    ],
)
def test_carrier_masks_preserve_exact_channel_exchange_symmetry(
    mask_name: str,
    swapped_name: str,
) -> None:
    rng = np.random.default_rng(20260724)
    windows = rng.uniform(-1.0, 1.0, size=(4, 8, 2))
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)

    reference, _ = evolve_carrier_mask_probabilities(
        windows,
        _reservoir(),
        geometry,
        np.asarray(CARRIER_MASKS[mask_name], dtype=float),
        mask_name=mask_name,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    mirrored, _ = evolve_carrier_mask_probabilities(
        windows[:, :, ::-1],
        _reservoir(),
        geometry,
        np.asarray(CARRIER_MASKS[swapped_name], dtype=float),
        mask_name=swapped_name,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    np.testing.assert_allclose(mirrored, reference, atol=1e-12, rtol=1e-12)


def test_architecture_representations_have_expected_widths() -> None:
    rng = np.random.default_rng(17)
    pure = {name: rng.normal(size=(10, 63)) for name in BILINEAR_SCHEDULES}
    carrier = {name: rng.normal(size=(10, 63)) for name in CARRIER_MASKS}
    representations = build_architecture_representations(pure, carrier)

    assert representations["pure_slot_all_orders"].shape == (10, 252)
    assert representations["carrier_identity_pair"].shape == (10, 126)
    assert representations["carrier_hadamard_pair"].shape == (10, 126)
    assert representations["carrier_all_masks"].shape == (10, 252)


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
            "capacity": [0.21, 0.04, 0.03, 0.8],
        }
    )
    result = _mixing_decomposition(metrics)
    assert result["same_lag_mixing"] == pytest.approx(0.21)
    assert result["delayed_mixing"] == pytest.approx(0.07)


def test_carrier_vs_pure_differences_are_paired_within_seed() -> None:
    rows = []
    for seed, pure, carrier in ((1, 0.02, 0.05), (2, 0.06, 0.01)):
        for representation, value in (
            ("pure_slot_all_orders", pure),
            ("carrier_all_masks", carrier),
        ):
            rows.append(
                {
                    "seed": seed,
                    "interaction": "on",
                    "representation": representation,
                    "minimum_early": 0.4 + value,
                    "minimum_delay5": 0.1 + value,
                    "mixing_sum": value,
                    "same_lag_mixing": value,
                    "delayed_mixing": value / 2.0,
                    "order": 0.2 + value,
                }
            )
    result = _carrier_vs_pure(pd.DataFrame(rows)).sort_values("seed")
    np.testing.assert_allclose(
        result["same_lag_mixing_carrier_minus_pure"].to_numpy(dtype=float),
        np.asarray([0.03, -0.05]),
    )
    np.testing.assert_allclose(
        result["delayed_mixing_carrier_minus_pure"].to_numpy(dtype=float),
        np.asarray([0.015, -0.025]),
    )
