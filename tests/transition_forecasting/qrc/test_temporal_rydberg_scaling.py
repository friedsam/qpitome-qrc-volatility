from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.temporal_rydberg_scaling import (
    TemporalRydbergScalingConfig,
    aggregate_scaling_summary,
    exact_state_memory_mib,
)


def test_scaling_config_is_bounded_and_ordered() -> None:
    config = TemporalRydbergScalingConfig()
    config.validate()
    assert config.n_atoms == (5, 6, 8, 10)
    assert config.conditions == ("ordered", "shuffled", "interaction_off")
    assert config.max_per_class == 2


def test_scaling_config_rejects_unsorted_or_missing_ordered() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TemporalRydbergScalingConfig(n_atoms=(6, 5)).validate()
    with pytest.raises(ValueError, match="ordered must be included"):
        TemporalRydbergScalingConfig(conditions=("shuffled",)).validate()


def test_exact_state_memory_doubles_per_qubit() -> None:
    memory_5 = exact_state_memory_mib(5)
    memory_6 = exact_state_memory_mib(6)
    assert memory_6 == pytest.approx(2.0 * memory_5)
    assert exact_state_memory_mib(6, samples=3) == pytest.approx(3.0 * memory_6)


def test_aggregate_scaling_summary_merges_mechanism_sensitivity() -> None:
    condition_metrics = pd.DataFrame(
        [
            {
                "n_atoms": 5,
                "condition": "ordered",
                "state_dimension": 32,
                "state_memory_mib_per_sample": 0.001,
                "state_memory_mib_per_batch": 0.002,
                "samples": 8,
                "feature_count": 30,
                "effective_rank_train": 3.0,
                "numerical_rank_train": 4,
                "effective_rank_fraction": 0.75,
                "mean_feature_std": 0.1,
                "wall_seconds": 2.0,
                "seconds_per_sample": 0.25,
            },
            {
                "n_atoms": 6,
                "condition": "ordered",
                "state_dimension": 64,
                "state_memory_mib_per_sample": 0.002,
                "state_memory_mib_per_batch": 0.004,
                "samples": 8,
                "feature_count": 36,
                "effective_rank_train": 3.5,
                "numerical_rank_train": 4,
                "effective_rank_fraction": 0.875,
                "mean_feature_std": 0.11,
                "wall_seconds": 4.0,
                "seconds_per_sample": 0.5,
            },
        ]
    )
    comparisons = pd.DataFrame(
        [
            {
                "n_atoms": 5,
                "comparison": "ordered_minus_shuffled",
                "mean_absolute_difference": 0.03,
            },
            {
                "n_atoms": 5,
                "comparison": "ordered_minus_interaction_off",
                "mean_absolute_difference": 0.07,
            },
            {
                "n_atoms": 6,
                "comparison": "ordered_minus_shuffled",
                "mean_absolute_difference": 0.04,
            },
            {
                "n_atoms": 6,
                "comparison": "ordered_minus_interaction_off",
                "mean_absolute_difference": 0.08,
            },
        ]
    )

    summary = aggregate_scaling_summary(condition_metrics, comparisons)
    assert summary["n_atoms"].tolist() == [5, 6]
    assert np.allclose(summary["order_sensitivity_mad"], [0.03, 0.04])
    assert np.allclose(summary["interaction_sensitivity_mad"], [0.07, 0.08])
