from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.bivariate_capacity_assay import (
    BivariateCapacityConfig,
    _evolve_segment_batch,
    build_capacity_targets,
    build_feature_banks,
    fit_capacity_readout,
    generate_bivariate_windows,
    probabilities_to_full_low_order,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _evolve_step_batch,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)


def test_phase_zero_general_segment_matches_incumbent_evolution() -> None:
    rng = np.random.default_rng(20260724)
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        step_duration_us=0.03,
        probe_fractions=(1.0,),
        shots=None,
    )
    geometry = StaggeredLadderGeometryConfig()
    precomputed = precompute_ladder(reservoir, geometry, interaction_scale=1.25)
    states = rng.normal(size=(5, 64)) + 1j * rng.normal(size=(5, 64))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    omega = rng.uniform(3.0, 8.0, size=5)
    delta = rng.uniform(1.0, 10.0, size=5)

    incumbent = _evolve_step_batch(
        states.copy(),
        omega,
        delta,
        reservoir,
        precomputed,
        interactions=True,
    )
    generalized = _evolve_segment_batch(
        states.copy(),
        omega,
        delta,
        np.zeros(5),
        reservoir.step_duration_us,
        reservoir,
        precomputed,
        interactions=True,
    )

    np.testing.assert_allclose(generalized, incumbent, atol=1e-12, rtol=1e-12)


def test_feature_banks_have_expected_three_probe_widths() -> None:
    probabilities = np.full((4, 3, 64), 1.0 / 64.0)
    banks = build_feature_banks(probabilities)

    assert banks["six_mode_density_curvature"].shape == (4, 6)
    assert banks["nine_mode_symmetric"].shape == (4, 9)
    assert banks["all_occupations"].shape == (4, 18)
    assert banks["full_one_two_body"].shape == (4, 108)


def test_connected_correlations_vanish_for_product_distribution() -> None:
    states = np.arange(64)
    bits = np.stack(
        [((states >> (5 - site)) & 1) for site in range(6)], axis=1
    ).astype(float)
    site_probability = np.asarray([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
    probability = np.prod(
        site_probability[None, :] ** bits
        * (1.0 - site_probability[None, :]) ** (1.0 - bits),
        axis=1,
    )
    probability /= probability.sum()
    probes = np.broadcast_to(probability, (1, 2, 64)).copy()

    full, names = probabilities_to_full_low_order(probes)
    connected_indices = [
        index for index, name in enumerate(names) if "_connected_" in name
    ]

    assert len(connected_indices) == 30
    np.testing.assert_allclose(full[:, connected_indices], 0.0, atol=1e-12)


def test_capacity_targets_include_memory_mixing_and_order() -> None:
    config = BivariateCapacityConfig(
        samples=100,
        sequence_length=16,
        memory_delays=(1, 2, 5),
    )
    windows = generate_bivariate_windows(config)
    targets = build_capacity_targets(windows, config)

    assert targets.values.shape == (100, 12)
    assert set(targets.metadata["group"]) == {"memory", "mixing", "order"}
    order_index = targets.metadata.index[
        targets.metadata["task"].eq("antisymmetric_order_d1_d2")
    ].item()
    expected = (
        windows[:, -2, 0] * windows[:, -3, 1]
        - windows[:, -2, 1] * windows[:, -3, 0]
    )
    np.testing.assert_allclose(targets.values[:, order_index], expected)


def test_raw_linear_readout_recovers_memory_but_not_products() -> None:
    config = BivariateCapacityConfig(
        samples=500,
        sequence_length=16,
        memory_delays=(1, 2, 5),
        alphas=(1e-6, 1e-4, 1e-2),
        seed=91,
    )
    windows = generate_bivariate_windows(config)
    targets = build_capacity_targets(windows, config)
    task_metrics, _, _ = fit_capacity_readout(
        windows.reshape(len(windows), -1), targets, config
    )

    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    nonlinear = task_metrics.loc[task_metrics["group"].isin(["mixing", "order"])]
    assert float(memory["corr2"].min()) > 0.999
    assert float(nonlinear["corr2"].mean()) < 0.20
