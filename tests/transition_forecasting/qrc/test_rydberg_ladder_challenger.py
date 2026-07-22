from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.rydberg_ladder_challenger_tools import (
    LadderChallengerConfig,
    architecture_cases,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    build_temporal_rydberg_chain_features,
)
from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import (
    feature_names_from_metadata,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    build_temporal_rydberg_ladder_features,
    ladder_pair_groups,
    staggered_ladder_positions,
)


def test_staggered_ladder_positions_are_centered_unique_and_asymmetric() -> None:
    positions = staggered_ladder_positions(StaggeredLadderGeometryConfig())
    assert positions.shape == (6, 2)
    assert np.allclose(positions.mean(axis=0), 0.0)
    distances = np.sqrt(
        np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
    )
    positive = distances[np.triu_indices(6, k=1)]
    assert np.all(positive > 0)
    assert len(np.unique(np.round(positive, decimals=6))) > 5


def test_ladder_pair_groups_are_valid_and_unique_inside_each_group() -> None:
    groups = ladder_pair_groups()
    assert set(groups) == {"row", "rung", "diagonal", "long"}
    for pairs in groups.values():
        normalized = [tuple(sorted(pair)) for pair in pairs]
        assert len(normalized) == len(set(normalized))
        assert all(0 <= left < right < 6 for left, right in normalized)


def test_architecture_cases_include_baselines_sweep_and_controls() -> None:
    config = LadderChallengerConfig(
        ladder_interaction_scales=(0.75, 1.0),
        controls=("reset", "reversed"),
    )
    names = [case.name for case in architecture_cases(config)]
    assert names == [
        "chain_interaction_off",
        "chain_interacting_1p00",
        "ladder_interaction_off",
        "ladder_ordered_0p75",
        "ladder_ordered_1p00",
        "ladder_reset_1p00",
        "ladder_reversed_1p00",
    ]


def test_noninteracting_chain_and_ladder_have_equal_occupations() -> None:
    rng = np.random.default_rng(7)
    windows = rng.uniform(-0.5, 0.5, size=(3, 5, 2))
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        step_duration_us=0.01,
        probe_fractions=(0.5, 1.0),
    )
    chain_features, chain_metadata = build_temporal_rydberg_chain_features(
        windows,
        reservoir,
        condition="interaction_off",
    )
    ladder_features, ladder_metadata = build_temporal_rydberg_ladder_features(
        windows,
        reservoir,
        StaggeredLadderGeometryConfig(),
        interaction_scale=0.0,
        condition="interaction_off",
    )
    chain_names = feature_names_from_metadata(chain_metadata)
    ladder_names = feature_names_from_metadata(ladder_metadata)
    chain_indices = [
        index
        for index, name in enumerate(chain_names)
        if "occupation_site_" in name
    ]
    ladder_indices = [
        index
        for index, name in enumerate(ladder_names)
        if "occupation_site_" in name
    ]
    assert np.allclose(
        chain_features[:, chain_indices],
        ladder_features[:, ladder_indices],
        atol=1e-12,
        rtol=1e-12,
    )
