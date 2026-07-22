from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.instability_mechanism_tools import (
    InstabilityMechanismAssayConfig,
    case_specs,
    family_indices,
    reorder_level,
)
from transition_forecasting.qrc.representation_candidates import (
    SECOND_CHANNEL_NAMES,
)
from transition_forecasting.qrc.rydberg_instability_mechanism_assay import (
    ASSAY_REPRESENTATION,
)


def test_case_specs_include_ordered_sweep_and_selected_controls() -> None:
    config = InstabilityMechanismAssayConfig(
        interaction_scales=(0.0, 0.5, 1.0),
        control_scales=(0.5,),
        controls=("reset", "reversed"),
    )
    assert case_specs(config) == (
        ("ordered", 0.0),
        ("ordered", 0.5),
        ("ordered", 1.0),
        ("reset", 0.5),
        ("reversed", 0.5),
    )


def test_assay_uses_registered_instability_representation() -> None:
    assert ASSAY_REPRESENTATION == "level_instability"
    assert SECOND_CHANNEL_NAMES[ASSAY_REPRESENTATION] == "local_instability_5"


def test_order_controls_are_deterministic_and_preserve_values() -> None:
    level = np.arange(2 * 10, dtype=float).reshape(2, 10)
    shuffled_a = reorder_level(level, "shuffled", seed=7, block_size=5)
    shuffled_b = reorder_level(level, "shuffled", seed=7, block_size=5)
    assert np.array_equal(shuffled_a, shuffled_b)
    assert np.array_equal(np.sort(shuffled_a, axis=1), np.sort(level, axis=1))
    assert np.array_equal(
        reorder_level(level, "reversed", seed=1, block_size=5),
        level[:, ::-1],
    )


def test_block_shuffle_preserves_values_inside_each_original_block() -> None:
    level = np.arange(12, dtype=float)[None, :]
    result = reorder_level(level, "block_shuffled", seed=4, block_size=3)
    blocks = {
        tuple(level[0, start : start + 3])
        for start in range(0, 12, 3)
    }
    result_blocks = {
        tuple(result[0, start : start + 3])
        for start in range(0, 12, 3)
    }
    assert result_blocks == blocks


def test_observable_families_select_expected_columns() -> None:
    names = (
        "probe_10_occupation_site_0",
        "probe_10_nearest_pair_0_1",
        "probe_10_nearest_connected_0_1",
        "probe_10_excitation_density",
        "probe_20_long_pair_0_5",
        "probe_20_long_connected_0_5",
        "probe_20_mean_domain_wall",
    )
    assert family_indices(names, "occupation").tolist() == [0]
    assert family_indices(names, "pair").tolist() == [1, 4]
    assert family_indices(names, "connected").tolist() == [2, 5]
    assert family_indices(names, "summary").tolist() == [3, 6]
    assert family_indices(names, "probe_10").tolist() == [0, 1, 2, 3]
