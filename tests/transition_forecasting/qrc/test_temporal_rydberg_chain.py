from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    alternating_chain_positions,
    build_temporal_rydberg_chain_features,
    controlled_level_windows,
    fit_level_rate_scaler,
    raw_rate_from_level,
    transform_level_windows,
)


def test_chain_geometry_is_asymmetric_and_alternating() -> None:
    config = TemporalRydbergChainConfig()
    positions = alternating_chain_positions(config)
    gaps = np.diff(positions[:, 0])

    assert positions.shape == (config.n_atoms, 2)
    assert np.all(gaps > 0)
    assert not np.allclose(gaps, gaps[::-1])
    assert np.isclose(gaps[0], config.spacing_short_um)
    assert np.isclose(
        gaps[1],
        config.spacing_long_um + config.defect_offset_um,
    )


def test_rate_is_recomputed_after_temporal_control() -> None:
    level = np.asarray(
        [
            [1.0, 2.0, 4.0, 8.0],
            [0.0, 1.0, 0.0, -1.0],
        ]
    )
    shuffled = controlled_level_windows(
        level,
        "shuffled",
        seed=7,
    )
    mask = np.asarray([True, False])
    scaler = fit_level_rate_scaler(level, mask)
    transformed = transform_level_windows(shuffled, scaler)

    expected_rate = raw_rate_from_level(shuffled)
    expected_scaled = np.clip(
        (expected_rate - scaler.rate_median)
        / scaler.rate_half_range,
        -1.0,
        1.0,
    )
    assert np.allclose(
        transformed[:, :, 1],
        expected_scaled,
    )


def test_time_channel_is_not_required() -> None:
    level = np.asarray(
        [
            [0.0, 0.2, 0.1, 0.5],
            [0.1, 0.0, -0.1, 0.2],
        ]
    )
    mask = np.asarray([True, False])
    scaler = fit_level_rate_scaler(level, mask)
    windows = transform_level_windows(level, scaler)

    assert windows.shape == (2, 4, 2)
    assert np.allclose(windows[:, 0, 1], 0.0)


def test_temporal_features_are_finite_and_controls_differ() -> None:
    level = np.asarray(
        [
            np.linspace(-1.0, 1.0, 8),
            np.asarray(
                [0.0, 0.3, -0.2, 0.6, 0.1, 0.7, 0.2, 0.9]
            ),
            np.asarray(
                [0.4, 0.2, 0.1, -0.1, -0.3, -0.2, 0.0, 0.2]
            ),
        ]
    )
    mask = np.asarray([True, True, False])
    scaler = fit_level_rate_scaler(level, mask)
    ordered_windows = transform_level_windows(level, scaler)
    reversed_windows = transform_level_windows(
        level[:, ::-1],
        scaler,
    )
    config = TemporalRydbergChainConfig(
        n_atoms=4,
        defect_edge=1,
        step_duration_us=0.03,
        probe_fractions=(0.5, 1.0),
    )

    ordered, ordered_meta = build_temporal_rydberg_chain_features(
        ordered_windows,
        config,
        condition="ordered",
    )
    reversed_features, _ = build_temporal_rydberg_chain_features(
        reversed_windows,
        config,
        condition="reversed",
    )
    reset, _ = build_temporal_rydberg_chain_features(
        ordered_windows,
        config,
        condition="reset",
    )
    interaction_off, _ = build_temporal_rydberg_chain_features(
        ordered_windows,
        config,
        condition="interaction_off",
    )

    assert (
        ordered.shape
        == reversed_features.shape
        == reset.shape
        == interaction_off.shape
    )
    assert np.isfinite(ordered).all()
    assert ordered_meta["probe_steps"] == [4, 8]
    assert not np.allclose(ordered, reversed_features)
    assert not np.allclose(ordered, reset)
    assert not np.allclose(ordered, interaction_off)
