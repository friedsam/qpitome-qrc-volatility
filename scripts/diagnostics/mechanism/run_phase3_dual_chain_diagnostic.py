#!/usr/bin/env python
"""Dual-chain geometry diagnostic for the Rydberg reservoir.

Measures:
1. slow/fast sublattice occupation response over anchors
2. within-chain and cross-chain connected correlations
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qpitome_qrc.qrc.rydberg_reservoir import (
    AquilaConstraints,
    RydbergQRCConfig,
    _drive_from_window,
    _evolve_segment_batch,
    _measure_features,
    precompute,
    select_rydberg_anchor_indices,
)


def connected_pair_summary(features, pre, n_slow):
    n = pre.n_atoms
    occ = features[:, :n]
    pairs = features[:, n:]

    pair_index = pre.pair_index

    slow_vals = []
    fast_vals = []
    cross_vals = []

    for j, (a, b) in enumerate(pair_index):
        conn = pairs[:, j] - occ[:, a] * occ[:, b]
        val = float(np.mean(np.abs(conn)))

        if a < n_slow and b < n_slow:
            slow_vals.append(val)
        elif a >= n_slow and b >= n_slow:
            fast_vals.append(val)
        else:
            cross_vals.append(val)

    return {
        "within_slow_abs_connected": float(np.mean(slow_vals)) if slow_vals else 0.0,
        "within_fast_abs_connected": float(np.mean(fast_vals)) if fast_vals else 0.0,
        "cross_abs_connected": float(np.mean(cross_vals)) if cross_vals else 0.0,
    }


def main():
    outdir = Path("scratch/dual_chain_diagnostic")
    outdir.mkdir(parents=True, exist_ok=True)

    config = RydbergQRCConfig(
        memory_mode="temporal",
        geometry="dual_chain",
        observable_mode="n_nn",
        collect_anchor_features=True,
        total_time_us=0.55,
        anchor_count=8,
        anchor_policy="even",
        reverse_anchors=True,
        encoding="plateau",
    )
    hw = AquilaConstraints()
    pre = precompute(config)

    rng = np.random.default_rng(123)

    # Synthetic market-like probes:
    # level carries slow drift; rate carries alternating fast shocks.
    S = 512
    T = config.lookback_days
    x = np.linspace(-1.0, 1.0, T)

    windows = np.zeros((S, T, 2), dtype=float)
    amplitudes = rng.uniform(-1.0, 1.0, size=S)
    shocks = rng.choice([-1.0, 1.0], size=(S, T))

    windows[:, :, 0] = amplitudes[:, None] * x[None, :]
    windows[:, :, 1] = 0.65 * shocks

    anchor_indices = select_rydberg_anchor_indices(config)
    K = len(anchor_indices)
    t_seg = config.total_time_us / K
    deltas, omegas = _drive_from_window(windows, anchor_indices, config, hw)

    dim = 2 ** pre.n_atoms
    states = np.zeros((S, dim), dtype=complex)
    states[:, 0] = 1.0

    slow_occ = []
    fast_occ = []
    conn_slow = []
    conn_fast = []
    conn_cross = []

    n_slow = config.n_atoms_slow

    for k in range(K):
        states = _evolve_segment_batch(
            states, omegas[:, k], deltas[:, k], t_seg, pre, config
        )
        feats = _measure_features(states, pre, config, rng=None)

        occ = feats[:, :pre.n_atoms]
        slow_occ.append(float(occ[:, :n_slow].mean()))
        fast_occ.append(float(occ[:, n_slow:].mean()))

        c = connected_pair_summary(feats, pre, n_slow)
        conn_slow.append(c["within_slow_abs_connected"])
        conn_fast.append(c["within_fast_abs_connected"])
        conn_cross.append(c["cross_abs_connected"])

    summary = {
        "anchor_indices": anchor_indices.tolist(),
        "slow_occ": slow_occ,
        "fast_occ": fast_occ,
        "within_slow_abs_connected": conn_slow,
        "within_fast_abs_connected": conn_fast,
        "cross_abs_connected": conn_cross,
        "mean_within_slow_abs_connected": float(np.mean(conn_slow)),
        "mean_within_fast_abs_connected": float(np.mean(conn_fast)),
        "mean_cross_abs_connected": float(np.mean(conn_cross)),
        "slow_fast_connected_ratio": float(np.mean(conn_slow) / max(np.mean(conn_fast), 1e-12)),
        "cross_to_within_connected_ratio": float(
            np.mean(conn_cross) / max(0.5 * (np.mean(conn_slow) + np.mean(conn_fast)), 1e-12)
        ),
    }

    json_path = outdir / "dual_chain_diagnostic.json"
    json_path.write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots()
    ax.plot(range(1, K + 1), slow_occ, marker="o", label="slow chain occupation")
    ax.plot(range(1, K + 1), fast_occ, marker="o", label="fast chain occupation")
    ax.set_xlabel("anchor segment")
    ax.set_ylabel("mean occupation")
    ax.set_title("Dual-chain occupation response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "dual_chain_occupation.png", dpi=180)

    fig, ax = plt.subplots()
    ax.plot(range(1, K + 1), conn_slow, marker="o", label="within slow")
    ax.plot(range(1, K + 1), conn_fast, marker="o", label="within fast")
    ax.plot(range(1, K + 1), conn_cross, marker="o", label="cross slow-fast")
    ax.set_xlabel("anchor segment")
    ax.set_ylabel("mean |connected correlation|")
    ax.set_title("Dual-chain connected correlations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "dual_chain_connected_correlations.png", dpi=180)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
