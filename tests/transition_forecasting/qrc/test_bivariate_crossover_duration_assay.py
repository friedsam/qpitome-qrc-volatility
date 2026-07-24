from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from transition_forecasting.qrc.bivariate_crossover_assay import (
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.bivariate_crossover_duration_assay import (
    BivariateCrossoverDurationConfig,
    PALINDROMIC_SCHEDULE_NAME,
    _aggregate,
    _schedule,
)
from transition_forecasting.qrc.bivariate_crossover_stability_assay import (
    generate_exchange_symmetric_windows,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def _config() -> BivariateCrossoverDurationConfig:
    return BivariateCrossoverDurationConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        step_durations_us=(0.01, 0.03),
        permutations=4,
    )


def test_duration_configuration_preserves_exchange_pairs() -> None:
    config = _config()
    config.validate()
    assert config.samples % 2 == 0
    assert int(np.floor(config.samples * 0.60)) % 2 == 0
    assert (
        int(np.floor(config.samples * 0.60))
        + int(np.floor(config.samples * 0.20))
    ) % 2 == 0


def test_duration_sweep_uses_palindromic_crossover() -> None:
    schedule = _schedule()
    assert schedule.name == PALINDROMIC_SCHEDULE_NAME
    assert schedule.segments == (("A", 0.25), ("B", 0.5), ("A", 0.25))


def test_duration_changes_dynamics_but_not_feature_width() -> None:
    config = _config()
    stability = config.stability_config()
    windows = generate_exchange_symmetric_windows(stability, seed=20260724)[:4]
    base = TemporalRydbergChainConfig(
        n_atoms=6,
        step_duration_us=0.03,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)

    short_probabilities, short_metadata = evolve_crossover_probabilities(
        windows,
        replace(base, step_duration_us=0.01),
        geometry,
        _schedule(),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    long_probabilities, long_metadata = evolve_crossover_probabilities(
        windows,
        replace(base, step_duration_us=0.03),
        geometry,
        _schedule(),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )

    short_features = build_crossover_feature_banks(short_probabilities)[
        "occupation_pair_raw"
    ]
    long_features = build_crossover_feature_banks(long_probabilities)[
        "occupation_pair_raw"
    ]
    assert short_features.shape == long_features.shape == (4, 42)
    assert not np.allclose(short_features, long_features)
    assert np.isclose(short_metadata["total_evolution_time_us"], 0.20)
    assert np.isclose(long_metadata["total_evolution_time_us"], 0.60)


def test_aggregate_keeps_worst_seed_balance_and_null_fractions() -> None:
    frame = pd.DataFrame(
        {
            "seed": [1, 2, 1, 2],
            "step_duration_us": [0.01, 0.01, 0.03, 0.03],
            "channel1_early": [0.2, 0.4, 0.3, 0.5],
            "channel2_early": [0.2, 0.2, 0.3, 0.5],
            "minimum_early": [0.2, 0.2, 0.3, 0.5],
            "early_balance_ratio": [1.0, 0.5, 1.0, 1.0],
            "channel1_delay5": [0.1, 0.1, 0.2, 0.2],
            "channel2_delay5": [0.1, 0.05, 0.2, 0.2],
            "minimum_delay5": [0.1, 0.05, 0.2, 0.2],
            "mixing_sum": [0.0, 0.0, 0.02, 0.03],
            "mixing_margin": [-0.02, -0.01, 0.01, 0.02],
            "order": [0.0, 0.0, 0.0, 0.0],
            "effective_rank_train_validation": [4.0, 4.0, 5.0, 5.0],
            "both_channels_early_above_null": [True, False, True, True],
            "minimum_delay5_above_null_q95": [True, False, True, True],
            "mixing_sum_above_null_q95": [False, False, True, True],
        }
    )
    aggregate = _aggregate(frame).set_index("step_duration_us")
    assert np.isclose(aggregate.loc[0.01, "early_balance_ratio_min"], 0.5)
    assert np.isclose(
        aggregate.loc[0.01, "both_channels_early_above_null_fraction"], 0.5
    )
    assert np.isclose(aggregate.loc[0.03, "minimum_delay5_mean"], 0.2)
    assert np.isclose(aggregate.loc[0.03, "mixing_above_null_fraction"], 1.0)
