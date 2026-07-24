from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_bilinear_mixing_assay import (
    BILINEAR_SCHEDULES,
)
from transition_forecasting.qrc.bivariate_carrier_area_match_assay import (
    AREA_MATCH_SCALE,
    BivariateCarrierAreaMatchConfig,
    _architecture_differences,
    area_matched_reservoir,
    build_area_match_representations,
    materialize_carrier_source_run,
)
from transition_forecasting.qrc.bivariate_carrier_crossmix_assay import (
    CARRIER_MASKS,
    carrier_drive,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)


def _reservoir() -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.015,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=20260724,
    )


def test_area_match_configuration_is_fixed_not_a_scale_sweep() -> None:
    config = BivariateCarrierAreaMatchConfig(
        seeds=(20260724,),
        permutations=4,
    )
    config.validate()
    assert config.field_scale == AREA_MATCH_SCALE

    with pytest.raises(ValueError, match="exactly 0.5"):
        BivariateCarrierAreaMatchConfig(field_scale=0.6).validate()


def test_area_matched_reservoir_halves_fields_but_not_duration() -> None:
    source = _reservoir()
    scaled = area_matched_reservoir(source, seed=20260725)

    assert scaled.omega_base_rad_us == pytest.approx(3.0)
    assert scaled.delta_center_rad_us == pytest.approx(3.0)
    assert scaled.delta_span_rad_us == pytest.approx(2.0)
    assert scaled.omega_mod_fraction == source.omega_mod_fraction
    assert scaled.step_duration_us == source.step_duration_us
    assert scaled.shot_seed == 20260725


def test_identity_carrier_matches_first_order_pure_slot_field_areas() -> None:
    source = _reservoir()
    scaled = area_matched_reservoir(source, seed=20260724)
    windows = np.asarray(
        [
            [[-0.7, 0.4]],
            [[0.2, -0.9]],
            [[0.8, 0.6]],
        ],
        dtype=float,
    )
    omega, delta, _ = carrier_drive(
        windows,
        0,
        np.asarray(CARRIER_MASKS["identity"], dtype=float),
        scaled,
        0.0,
    )
    total_time = source.step_duration_us
    pure_delta_area = (
        source.delta_center_rad_us
        + source.delta_span_rad_us * windows[:, 0, 0]
    ) * total_time * 0.5
    pure_omega_area = (
        source.omega_base_rad_us
        * (1.0 + source.omega_mod_fraction * windows[:, 0, 1])
        * total_time
        * 0.5
    )

    np.testing.assert_allclose(delta * total_time, pure_delta_area, atol=1e-14)
    np.testing.assert_allclose(omega * total_time, pure_omega_area, atol=1e-14)


def test_area_match_representations_have_expected_widths() -> None:
    rng = np.random.default_rng(91)
    pure = {
        name: rng.normal(size=(10, 63))
        for name in BILINEAR_SCHEDULES
    }
    full = {
        name: rng.normal(size=(10, 63))
        for name in CARRIER_MASKS
    }
    area = {
        name: rng.normal(size=(10, 63))
        for name in CARRIER_MASKS
    }

    representations = build_area_match_representations(pure, full, area)

    assert representations["pure_slot_all_orders"].shape == (10, 252)
    assert representations["carrier_full_strength_all_masks"].shape == (10, 252)
    assert representations["carrier_area_identity_pair"].shape == (10, 126)
    assert representations["carrier_area_hadamard_pair"].shape == (10, 126)
    assert representations["carrier_area_all_masks"].shape == (10, 252)


def test_architecture_differences_are_paired_within_seed() -> None:
    rows = []
    values = {
        "pure_slot_all_orders": (0.4, 0.10, 0.02),
        "carrier_full_strength_all_masks": (0.5, 0.08, 0.00),
        "carrier_area_all_masks": (0.6, 0.12, 0.05),
    }
    for seed in (1, 2):
        for representation, (early, delay5, same_lag) in values.items():
            rows.append(
                {
                    "seed": seed,
                    "interaction": "on",
                    "representation": representation,
                    "minimum_early": early + seed * 0.01,
                    "minimum_delay5": delay5,
                    "mixing_sum": same_lag,
                    "same_lag_mixing": same_lag,
                    "delayed_mixing": 0.0,
                    "order": 0.0,
                }
            )
    differences = _architecture_differences(pd.DataFrame(rows)).sort_values("seed")

    np.testing.assert_allclose(
        differences["same_lag_mixing__area_minus_pure"],
        np.asarray([0.03, 0.03]),
    )
    np.testing.assert_allclose(
        differences["minimum_delay5__area_minus_full"],
        np.asarray([0.04, 0.04]),
    )


def test_source_run_materialization_accepts_directory_and_zip(tmp_path: Path) -> None:
    run = tmp_path / "bivariate_carrier_crossmix_001"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"status": "bivariate_carrier_crossmix_complete"}) + "\n",
        encoding="utf-8",
    )

    with materialize_carrier_source_run(run) as resolved:
        assert resolved == run

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(run / "summary.json", "nested/run/summary.json")
    with materialize_carrier_source_run(archive) as resolved:
        assert resolved.name == "run"
        assert (resolved / "summary.json").exists()
