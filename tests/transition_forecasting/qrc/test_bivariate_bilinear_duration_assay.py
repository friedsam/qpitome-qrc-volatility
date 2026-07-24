from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_bilinear_duration_assay import (
    BivariateBilinearDurationConfig,
    _annotate_nulls,
    _paired_on_off,
)
from transition_forecasting.qrc.bivariate_bilinear_mixing_assay import (
    BILINEAR_SCHEDULES,
    evolve_bilinear_schedule_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def _reservoir(duration: float) -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=duration,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )


def test_duration_config_preserves_exchange_pairs() -> None:
    config = BivariateBilinearDurationConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        step_durations_us=(0.01, 0.02, 0.03),
        permutations=4,
    )
    config.validate()


def test_duration_config_rejects_duplicate_or_nonpositive_values() -> None:
    with pytest.raises(ValueError):
        BivariateBilinearDurationConfig(
            step_durations_us=(0.01, 0.01)
        ).validate()
    with pytest.raises(ValueError):
        BivariateBilinearDurationConfig(
            step_durations_us=(0.0, 0.01)
        ).validate()


def test_bilinear_probabilities_change_with_duration() -> None:
    rng = np.random.default_rng(20260724)
    windows = rng.uniform(-1.0, 1.0, size=(3, 8, 2))
    geometry = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    routes = BILINEAR_SCHEDULES["d1_then_x2"]

    short, _ = evolve_bilinear_schedule_probabilities(
        windows,
        _reservoir(0.01),
        geometry,
        routes,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )
    long, _ = evolve_bilinear_schedule_probabilities(
        windows,
        _reservoir(0.03),
        geometry,
        routes,
        interaction_scale=1.25,
        drive_phase_rad=0.0,
    )

    assert short.shape == long.shape == (3, 2, 64)
    assert float(np.max(np.abs(short - long))) > 1e-8


def test_null_annotation_is_duration_specific() -> None:
    observed = pd.DataFrame(
        [
            {
                "seed": 1,
                "step_duration_us": 0.01,
                "interaction": "on",
                "representation": "all_orders_joint",
                "channel1_early": 0.5,
                "channel2_early": 0.5,
                "minimum_early": 0.5,
                "minimum_delay5": 0.2,
                "mixing_sum": 0.1,
                "order": 0.05,
            },
            {
                "seed": 1,
                "step_duration_us": 0.02,
                "interaction": "on",
                "representation": "all_orders_joint",
                "channel1_early": 0.5,
                "channel2_early": 0.5,
                "minimum_early": 0.5,
                "minimum_delay5": 0.2,
                "mixing_sum": 0.1,
                "order": 0.05,
            },
        ]
    )
    rows = []
    for duration, baseline in ((0.01, 0.01), (0.02, 0.20)):
        for permutation in range(4):
            rows.append(
                {
                    "seed": 1,
                    "step_duration_us": duration,
                    "interaction": "on",
                    "representation": "all_orders_joint",
                    "permutation": permutation,
                    "channel1_early": baseline,
                    "channel2_early": baseline,
                    "minimum_early": baseline,
                    "minimum_delay5": baseline,
                    "mixing_sum": baseline,
                    "order": baseline,
                }
            )
    annotated = _annotate_nulls(observed, pd.DataFrame(rows))
    first = annotated.loc[np.isclose(annotated["step_duration_us"], 0.01)].iloc[0]
    second = annotated.loc[np.isclose(annotated["step_duration_us"], 0.02)].iloc[0]
    assert first["mixing_sum_above_null_q95"]
    assert not second["mixing_sum_above_null_q95"]


def test_paired_on_off_is_kept_per_duration() -> None:
    rows = []
    for duration in (0.01, 0.02):
        for interaction, offset in (("off", 0.0), ("on", duration)):
            rows.append(
                {
                    "seed": 1,
                    "step_duration_us": duration,
                    "interaction": interaction,
                    "representation": "palindrome_plus_all_orders",
                    "minimum_early": 0.1 + offset,
                    "minimum_delay5": 0.2 + offset,
                    "mixing_sum": 0.3 + offset,
                    "order": 0.4 + offset,
                }
            )
    paired = _paired_on_off(pd.DataFrame(rows))
    assert len(paired) == 2
    np.testing.assert_allclose(
        paired.sort_values("step_duration_us")["mixing_sum_on_minus_off"],
        [0.01, 0.02],
    )
