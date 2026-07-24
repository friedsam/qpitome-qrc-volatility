from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.serial_multichannel_ladder import (
    SerialMultichannelEncodingConfig,
    _apply_global_xy_batch,
    build_multichannel_sequences,
    evolve_serial_ladder_probe_probabilities,
    fit_three_channel_scaler,
    transform_three_channel_sequences,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _apply_global_rx_batch,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def test_level_rate_time_contains_explicit_clock() -> None:
    level = np.array(
        [
            [0.0, 0.2, 0.1, 0.4, 0.3],
            [0.1, 0.0, 0.3, 0.2, 0.5],
        ]
    )
    config = SerialMultichannelEncodingConfig()

    sequences = build_multichannel_sequences(level, "level_rate_time", config)

    assert sequences.shape == (2, 5, 3)
    np.testing.assert_allclose(sequences[:, :, 0], level)
    np.testing.assert_allclose(sequences[0, :, 2], np.linspace(-1.0, 1.0, 5))
    np.testing.assert_allclose(sequences[0, :, 2], sequences[1, :, 2])


def test_phase_zero_rotation_matches_existing_global_rx() -> None:
    rng = np.random.default_rng(20260724)
    states = rng.normal(size=(4, 8)) + 1j * rng.normal(size=(4, 8))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    angles = np.array([0.1, 0.2, -0.3, 0.5])

    expected = _apply_global_rx_batch(states.copy(), angles, 3)
    observed = _apply_global_xy_batch(
        states.copy(), angles, np.zeros(4, dtype=float), 3
    )

    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_serial_ladder_probabilities_are_normalized() -> None:
    rng = np.random.default_rng(13)
    level = rng.normal(scale=0.2, size=(3, 6))
    train = np.array([True, True, False])
    encoding = SerialMultichannelEncodingConfig()
    raw = build_multichannel_sequences(level, "level_rate_time", encoding)
    scaler = fit_three_channel_scaler(raw, train)
    scaled = transform_three_channel_sequences(raw, scaler)
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        step_duration_us=0.003,
        probe_fractions=(0.5, 1.0),
        max_phase_per_substep=1.0,
        max_substeps_per_step=32,
        shots=None,
    )

    probabilities, metadata = evolve_serial_ladder_probe_probabilities(
        scaled,
        "level_rate_time",
        reservoir,
        StaggeredLadderGeometryConfig(),
        encoding,
        interaction_scale=0.05,
        condition="ordered",
    )

    assert probabilities.shape == (3, 2, 64)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=1e-12)
    assert metadata["pulses_per_day"] == 3
    assert metadata["probe_steps"] == [3, 6]
