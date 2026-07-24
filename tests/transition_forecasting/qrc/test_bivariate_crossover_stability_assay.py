from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.bivariate_capacity_assay import (
    build_capacity_targets,
    fit_capacity_readout,
)
from transition_forecasting.qrc.bivariate_crossover_stability_assay import (
    BivariateCrossoverStabilityConfig,
    build_stability_representations,
    generate_exchange_symmetric_windows,
    paired_permutation_orders,
)


def _config() -> BivariateCrossoverStabilityConfig:
    return BivariateCrossoverStabilityConfig(
        samples=160,
        sequence_length=20,
        memory_delays=(1, 2, 4, 5, 8, 12),
        seeds=(20260724,),
        permutations=4,
    )


def test_exchange_symmetric_windows_are_interleaved_exact_pairs() -> None:
    config = _config()
    windows = generate_exchange_symmetric_windows(config, seed=20260724)

    assert windows.shape == (160, 20, 2)
    np.testing.assert_allclose(windows[1::2], windows[0::2, :, ::-1], atol=0.0, rtol=0.0)


def test_capacity_splits_preserve_complete_exchange_pairs() -> None:
    config = _config()
    config.validate()

    train_end = int(np.floor(config.samples * 0.60))
    validation_end = train_end + int(np.floor(config.samples * 0.20))

    assert train_end % 2 == 0
    assert validation_end % 2 == 0
    assert (config.samples - validation_end) % 2 == 0


def test_paired_null_orders_are_permutations_of_complete_pairs() -> None:
    config = _config()
    orders = paired_permutation_orders(
        config.samples,
        config.permutations,
        rng=np.random.default_rng(99),
    )

    assert len(orders) == config.permutations
    for order in orders:
        np.testing.assert_array_equal(np.sort(order), np.arange(config.samples))
        assert np.all(order[0::2] // 2 == order[1::2] // 2)
        assert np.all(np.abs(order[0::2] - order[1::2]) == 1)


def test_mirrored_concatenation_has_expected_hardware_natural_width() -> None:
    rng = np.random.default_rng(71)
    left = rng.normal(size=(160, 63))
    right = rng.normal(size=(160, 63))

    representations = build_stability_representations(left, right)

    assert representations["palindrome_A"].shape == (160, 63)
    assert representations["palindrome_B"].shape == (160, 63)
    assert representations["concatenated_mirrors"].shape == (160, 126)
    np.testing.assert_allclose(representations["concatenated_mirrors"][:, :63], left)
    np.testing.assert_allclose(representations["concatenated_mirrors"][:, 63:], right)


def test_raw_exchange_paired_baseline_has_symmetric_channel_capacity() -> None:
    config = _config()
    windows = generate_exchange_symmetric_windows(config, seed=20260724)
    capacity_config = config.capacity_config(20260724)
    targets = build_capacity_targets(windows, capacity_config)

    metrics, _, _ = fit_capacity_readout(
        windows.reshape(config.samples, -1),
        targets,
        capacity_config,
    )
    memory = metrics.loc[metrics["group"].eq("memory")]
    channel_1 = (
        memory.loc[memory["channel"].eq(1)]
        .sort_values("delay_a")["capacity"]
        .to_numpy(dtype=float)
    )
    channel_2 = (
        memory.loc[memory["channel"].eq(2)]
        .sort_values("delay_a")["capacity"]
        .to_numpy(dtype=float)
    )

    np.testing.assert_allclose(channel_1, channel_2, atol=1e-10, rtol=1e-10)
