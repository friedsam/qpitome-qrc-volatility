from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_bilinear_duration_assay import (
    BivariateBilinearDurationConfig,
    REPRESENTATIONS,
    _paired_on_off,
    select_duration_representations,
)


def _config() -> BivariateBilinearDurationConfig:
    return BivariateBilinearDurationConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        step_durations_us=(0.01, 0.02, 0.03),
        permutations=4,
    )


def test_smoke_configuration_is_pair_preserving_and_contains_reference_duration() -> None:
    config = _config()
    config.validate()

    train_end = int(np.floor(config.samples * 0.60))
    validation_end = train_end + int(np.floor(config.samples * 0.20))
    assert train_end % 2 == 0
    assert validation_end % 2 == 0
    assert 0.02 in config.step_durations_us


def test_duration_configuration_rejects_duplicate_or_zero_durations() -> None:
    with pytest.raises(ValueError, match="step durations"):
        BivariateBilinearDurationConfig(
            samples=160,
            sequence_length=20,
            memory_delays=(1, 2, 4, 5, 8, 12),
            seeds=(20260724,),
            step_durations_us=(0.01, 0.01),
            permutations=4,
        ).validate()
    with pytest.raises(ValueError, match="step durations"):
        BivariateBilinearDurationConfig(
            samples=160,
            sequence_length=20,
            memory_delays=(1, 2, 4, 5, 8, 12),
            seeds=(20260724,),
            step_durations_us=(0.0, 0.02),
            permutations=4,
        ).validate()


def test_duration_representation_selection_is_bounded() -> None:
    rng = np.random.default_rng(11)
    values = {
        "palindrome_control": rng.normal(size=(10, 63)),
        "forward_mirror_concat": rng.normal(size=(10, 126)),
        "reverse_mirror_concat": rng.normal(size=(10, 126)),
        "commutator_contrast_concat": rng.normal(size=(10, 126)),
    }
    selected = select_duration_representations(values)

    assert tuple(selected) == REPRESENTATIONS
    assert selected["palindrome_control"].shape == (10, 63)
    assert selected["reverse_mirror_concat"].shape == (10, 126)
    assert selected["commutator_contrast_concat"].shape == (10, 126)
    assert "forward_mirror_concat" not in selected


def test_paired_on_off_differences_are_computed_within_seed() -> None:
    rows = []
    for seed, off, on in ((1, 0.01, 0.04), (2, 0.03, 0.02)):
        for interaction, value in (("off", off), ("on", on)):
            rows.append(
                {
                    "seed": seed,
                    "step_duration_us": 0.02,
                    "representation": "reverse_mirror_concat",
                    "interaction": interaction,
                    "mixing_sum": value,
                    "minimum_delay5": 0.1 + value,
                    "minimum_early": 0.2 + value,
                    "order": 0.3 + value,
                }
            )
    paired = _paired_on_off(pd.DataFrame(rows)).sort_values("seed")

    np.testing.assert_allclose(
        paired["mixing_sum_on_minus_off"].to_numpy(dtype=float),
        np.asarray([0.03, -0.01]),
    )
    np.testing.assert_allclose(
        paired["minimum_delay5_on_minus_off"].to_numpy(dtype=float),
        np.asarray([0.03, -0.01]),
    )
