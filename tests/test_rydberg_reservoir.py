"""Tests for the true temporal Rydberg reservoir.

Physics sanity first (Rabi, blockade, unitarity), then reservoir semantics
(memory and order controls actually change features), then hardware validator.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpitome_qrc.qrc.rydberg_reservoir import (
    AquilaConstraints,
    RydbergQRCConfig,
    _evolve_segment_batch,
    build_rydberg_feature_matrix,
    precompute,
    validate_aquila_feasibility,
)


def _single_atom_config(**kw) -> RydbergQRCConfig:
    base = dict(
        geometry="chain",
        chain_atoms=1,
        lookback_days=4,
        anchor_count=1,
        total_time_us=0.5,
        omega_base_rad_us=10.0,
        omega_mode="constant",
        delta_center_rad_us=0.0,
        delta_span_rad_us=0.0,
        collect_anchor_features=False,
    )
    base.update(kw)
    return RydbergQRCConfig(**base)


def test_single_atom_rabi_matches_analytic():
    """Resonant drive: <n>(t) = sin^2(Omega t / 2)."""
    omega, t = 10.0, 0.5
    config = _single_atom_config(omega_base_rad_us=omega, total_time_us=t)
    pre = precompute(config)
    states = np.zeros((1, 2), dtype=complex)
    states[0, 0] = 1.0
    states = _evolve_segment_batch(
        states, np.array([omega]), np.array([0.0]), t, pre, config
    )
    n_exp = float((np.abs(states[0]) ** 2) @ pre.occ_bits[:, 0])
    assert n_exp == pytest.approx(np.sin(omega * t / 2.0) ** 2, abs=1e-3)


def test_two_atom_blockade_suppresses_double_excitation():
    """Strong blockade (V >> Omega): <n1 n2> stays near zero."""
    config = RydbergQRCConfig(
        geometry="chain",
        chain_atoms=2,
        chain_spacing_um=5.0,  # V ~ 347 rad/us >> Omega
        lookback_days=4,
        anchor_count=1,
        total_time_us=0.6,
        omega_base_rad_us=6.0,
        omega_mode="constant",
        delta_center_rad_us=0.0,
        delta_span_rad_us=0.0,
    )
    pre = precompute(config)
    states = np.zeros((1, 4), dtype=complex)
    states[0, 0] = 1.0
    states = _evolve_segment_batch(
        states, np.array([6.0]), np.array([0.0]), 0.6, pre, config
    )
    probs = np.abs(states[0]) ** 2
    nn = float(probs @ pre.pair_bits[:, 0])
    assert nn < 0.03


def test_evolution_preserves_norm():
    config = RydbergQRCConfig(n_atoms_slow=3, n_atoms_fast=3, lookback_days=10, anchor_count=3)
    rng = np.random.default_rng(0)
    windows = rng.uniform(-1, 1, size=(5, 10, 2))
    pre = precompute(config)
    states = np.zeros((5, 2**pre.n_atoms), dtype=complex)
    states[:, 0] = 1.0
    states = _evolve_segment_batch(
        states, rng.uniform(4, 12, 5), rng.uniform(-5, 20, 5), 0.6, pre, config
    )
    norms = np.linalg.norm(states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)
    # And the full feature builder produces finite features in [0, 1].
    H = build_rydberg_feature_matrix(windows, config)
    assert np.all(np.isfinite(H))
    assert H.min() >= -1e-9 and H.max() <= 1.0 + 1e-9


def test_temporal_memory_and_order_matter():
    """Temporal vs memoryless differ; anchor shuffling changes features."""
    rng = np.random.default_rng(1)
    windows = rng.uniform(-1, 1, size=(4, 12, 2))
    base = dict(
        geometry="chain", chain_atoms=4, chain_spacing_um=7.0,
        lookback_days=12, anchor_count=4,
    )
    H_temporal = build_rydberg_feature_matrix(
        windows, RydbergQRCConfig(**base, memory_mode="temporal")
    )
    H_memoryless = build_rydberg_feature_matrix(
        windows, RydbergQRCConfig(**base, memory_mode="memoryless")
    )
    H_shuffled = build_rydberg_feature_matrix(
        windows, RydbergQRCConfig(**base, memory_mode="temporal", shuffle_anchors=True)
    )
    assert H_temporal.shape == H_memoryless.shape == H_shuffled.shape
    assert not np.allclose(H_temporal, H_memoryless, atol=1e-6)
    assert not np.allclose(H_temporal, H_shuffled, atol=1e-6)
    # Memoryless features per anchor must be identical for identical inputs:
    # feed a constant window and check anchor blocks repeat.
    const_windows = np.zeros((1, 12, 2))
    H_const = build_rydberg_feature_matrix(
        const_windows, RydbergQRCConfig(**base, memory_mode="memoryless")
    )
    n_feats = H_const.shape[1] // 4
    blocks = H_const.reshape(4, n_feats)
    assert np.allclose(blocks, blocks[0], atol=1e-9)


def test_shot_noise_converges_to_exact():
    rng = np.random.default_rng(2)
    windows = rng.uniform(-1, 1, size=(3, 12, 2))
    base = dict(
        geometry="chain", chain_atoms=4, chain_spacing_um=7.0,
        lookback_days=12, anchor_count=3,
    )
    H_exact = build_rydberg_feature_matrix(windows, RydbergQRCConfig(**base))
    H_shots = build_rydberg_feature_matrix(
        windows, RydbergQRCConfig(**base, shots=4000)
    )
    assert np.max(np.abs(H_exact - H_shots)) < 0.05


def test_feasibility_validator_flags_violations():
    ok = validate_aquila_feasibility(RydbergQRCConfig())
    assert ok["checks"]["omega_range_ok"]
    assert ok["checks"]["total_time_ok"]

    bad = validate_aquila_feasibility(
        RydbergQRCConfig(
            omega_base_rad_us=20.0,
            total_time_us=5.0,
            spacing_slow_um=2.0,
        ),
        AquilaConstraints(),
    )
    assert not bad["feasible"]
    assert not bad["checks"]["omega_range_ok"]
    assert not bad["checks"]["total_time_ok"]
    assert not bad["checks"]["min_spacing_ok"]


def test_ramp_encoding_matches_landau_zener():
    """Linear Delta sweep through resonance: excitation prob = 1 - exp(-pi*Omega^2/(2*alpha)).

    Our H = (Omega/2) sigma_x - Delta(t) n maps to the standard LZ problem with
    sweep rate alpha = |dDelta/dt| and gap Omega at the crossing.
    """
    from qpitome_qrc.qrc.rydberg_reservoir import _evolve_ramp_segment_batch

    omega = 3.0
    d_range = 40.0  # sweep -20 -> +20 rad/us through resonance at Delta=0
    config = _single_atom_config()
    pre = precompute(config)
    for t_total in (0.15, 2.0, 8.0):
        alpha = d_range / t_total
        expected = 1.0 - np.exp(-np.pi * omega**2 / (2.0 * alpha))
        states = np.zeros((1, 2), dtype=complex)
        states[0, 0] = 1.0
        states = _evolve_ramp_segment_batch(
            states, np.array([omega]), np.array([-20.0]), np.array([20.0]),
            t_total, pre, config,
        )
        n_exp = float((np.abs(states[0]) ** 2) @ pre.occ_bits[:, 0])
        assert n_exp == pytest.approx(expected, abs=0.06), (t_total, n_exp, expected)


def test_ramp_variants_run_and_differ():
    rng = np.random.default_rng(3)
    windows = rng.uniform(-1, 1, size=(4, 8, 2))
    base = dict(
        geometry="chain", chain_atoms=4, chain_spacing_um=9.0,
        lookback_days=8, anchor_count=8, total_time_us=0.8,
        omega_mode="constant", encoding="ramp",
    )
    H_t = build_rydberg_feature_matrix(windows, RydbergQRCConfig(**base))
    H_m = build_rydberg_feature_matrix(
        windows, RydbergQRCConfig(**base, memory_mode="memoryless")
    )
    H_s = build_rydberg_feature_matrix(
        windows, RydbergQRCConfig(**base, shuffle_anchors=True)
    )
    assert H_t.shape == H_m.shape == H_s.shape
    assert np.all(np.isfinite(H_t))
    assert not np.allclose(H_t, H_m, atol=1e-6)
    assert not np.allclose(H_t, H_s, atol=1e-6)
