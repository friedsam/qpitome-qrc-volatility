#!/usr/bin/env python3
"""Temporal many-body memory diagnostic (cross-time nonlinear capacity).

Distinguishes TRUE temporal memory from static nonlinear feature expansion
using controlled synthetic inputs (i.i.d. uniform(-1,1) anchor sequences),
in the spirit of Information Processing Capacity (Dambre et al., 2012).

Key fact: memoryless per-anchor features + a linear readout form an ADDITIVE
model over time, sum_k f(u_k). Such a model has, by construction, ZERO
capacity for cross-time product targets u_a * u_b (a != b), while it can
capture same-time nonlinearities like P2(u_k). Therefore:

  target class                 additive/static model    temporal memory
  degree-1 recall  u_k         yes (identity)           yes
  same-time  P2(u_k)           yes                      yes
  cross-time u_a*u_b, a!=b     NO (provably zero)       yes, if real

Capacity per target = max(0, 1 - MSE/Var) on HELD-OUT data with a ridge
readout. The money plot is cross-time capacity vs anchor separation |a-b|.

Variants:
  classical_additive   anchor values + squares (analytic zero reference)
  classical_products   + all pairwise products (analytic ceiling reference)
  plateau_memoryless   Rydberg static expansion control -> expect ~0 cross
  plateau_temporal     Rydberg plateau reservoir
  ramp_memoryless      sees adjacent (k-1, k) pairs -> expect sep-1 only
  ramp_temporal        Landau-Zener rate-sensing reservoir
  plateau_shuffled     order control (cross capacity present but scrambled)

All Rydberg variants use omega_mode="constant" so Delta(t) is the ONLY input
channel; otherwise the Omega rate channel would leak adjacent-difference
information into plateau variants and muddy interpretation.

Outputs to scratch/ (git-ignored). Promote intentionally.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    build_rydberg_feature_matrix,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ridge-alpha", type=float, default=1e-2)
    p.add_argument("--total-time-us", type=float, default=0.55)
    p.add_argument("--n-slow", type=int, default=4)
    p.add_argument("--n-fast", type=int, default=4)
    p.add_argument("--spacing-slow-um", type=float, default=9.0)
    p.add_argument("--spacing-fast-um", type=float, default=15.0)
    p.add_argument("--delta-center", type=float, default=6.0)
    p.add_argument("--delta-span", type=float, default=4.0)
    p.add_argument("--omega-base", type=float, default=6.0)
    p.add_argument("--out-dir", type=Path, default=Path("scratch/temporal_memory_diagnostic"))
    p.add_argument("--tag", default="v1")
    return p.parse_args()


def make_inputs(n: int, anchors: int, rng: np.random.Generator) -> np.ndarray:
    """(n, anchors, 2) windows: level = iid U(-1,1); rate channel unused."""
    u = rng.uniform(-1.0, 1.0, size=(n, anchors))
    rate = np.zeros_like(u)
    rate[:, 1:] = np.diff(u, axis=1)
    return np.stack([u, rate], axis=2)


def capacity(H_tr, y_tr, H_te, y_te, alpha: float) -> float:
    scaler = StandardScaler().fit(H_tr)
    model = Ridge(alpha=alpha).fit(scaler.transform(H_tr), y_tr)
    pred = model.predict(scaler.transform(H_te))
    var = float(np.var(y_te))
    if var <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.mean((pred - y_te) ** 2) / var))


def targets_for(u: np.ndarray) -> dict[str, tuple[str, int, np.ndarray]]:
    """Return {name: (class, separation, values)} for one input matrix (n, K)."""
    K = u.shape[1]
    out: dict[str, tuple[str, int, np.ndarray]] = {}
    for k in range(K):
        out[f"u[{k}]"] = ("linear", 0, u[:, k])
        out[f"P2(u[{k}])"] = ("same_time_nl", 0, 0.5 * (3.0 * u[:, k] ** 2 - 1.0))
    for a in range(K - 1):
        for b in range(a + 1, K):
            out[f"u[{a}]*u[{b}]"] = ("cross_time", b - a, u[:, a] * u[:, b])
    return out


def classical_features(u: np.ndarray, products: bool) -> np.ndarray:
    feats = [u, u**2]
    if products:
        K = u.shape[1]
        feats.append(
            np.stack([u[:, a] * u[:, b] for a in range(K - 1) for b in range(a + 1, K)], axis=1)
        )
    return np.concatenate(feats, axis=1)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    K = args.anchors
    X_tr = make_inputs(args.n_train, K, rng)
    X_te = make_inputs(args.n_test, K, rng)
    u_tr, u_te = X_tr[:, :, 0], X_te[:, :, 0]

    base = RydbergQRCConfig(
        n_atoms_slow=args.n_slow,
        n_atoms_fast=args.n_fast,
        spacing_slow_um=args.spacing_slow_um,
        spacing_fast_um=args.spacing_fast_um,
        lookback_days=K,
        anchor_count=K,
        anchor_policy="even",
        total_time_us=args.total_time_us,
        delta_center_rad_us=args.delta_center,
        delta_span_rad_us=args.delta_span,
        omega_base_rad_us=args.omega_base,
        omega_mode="constant",
    )
    variants: dict[str, RydbergQRCConfig | str] = {
        "classical_additive": "classical_additive",
        "classical_products": "classical_products",
        "plateau_memoryless": replace(base, memory_mode="memoryless"),
        "plateau_temporal": base,
        "plateau_shuffled": replace(base, shuffle_anchors=True),
        "ramp_memoryless": replace(base, encoding="ramp", memory_mode="memoryless"),
        "ramp_temporal": replace(base, encoding="ramp"),
    }

    tgt_tr = targets_for(u_tr)
    tgt_te = targets_for(u_te)

    rows = []
    for name, cfg in variants.items():
        print(f"=== {name} ===")
        if cfg == "classical_additive":
            H_tr, H_te = classical_features(u_tr, False), classical_features(u_te, False)
        elif cfg == "classical_products":
            H_tr, H_te = classical_features(u_tr, True), classical_features(u_te, True)
        else:
            H_tr = build_rydberg_feature_matrix(X_tr, cfg)
            H_te = build_rydberg_feature_matrix(X_te, cfg)
        for tname, (tclass, sep, y_tr) in tgt_tr.items():
            _, _, y_te = tgt_te[tname]
            c = capacity(H_tr, y_tr, H_te, y_te, args.ridge_alpha)
            rows.append(
                {"variant": name, "target": tname, "target_class": tclass, "separation": sep, "capacity": c}
            )
        df_v = pd.DataFrame([r for r in rows if r["variant"] == name])
        agg = df_v.groupby("target_class")["capacity"].mean().round(3).to_dict()
        print("  mean capacity by class:", agg)

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / f"temporal_memory_diagnostic_{args.tag}.csv"
    df.to_csv(out_csv, index=False)

    cross = df[df.target_class == "cross_time"]
    pivot = cross.groupby(["variant", "separation"])["capacity"].mean().unstack()
    summary = {
        "tag": args.tag,
        "config": {k: v for k, v in vars(args).items() if k not in {"out_dir"}},
        "mean_capacity_by_class": df.groupby(["variant", "target_class"])["capacity"]
        .mean()
        .unstack()
        .round(4)
        .to_dict(),
        "cross_time_capacity_by_separation": pivot.round(4).to_dict(),
        "total_cross_time_capacity": cross.groupby("variant")["capacity"].sum().round(3).to_dict(),
    }
    out_json = args.out_dir / f"temporal_memory_diagnostic_{args.tag}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant in pivot.index:
        axes[0].plot(pivot.columns, pivot.loc[variant], marker="o", label=variant)
    axes[0].set_xlabel("anchor separation |a - b|")
    axes[0].set_ylabel("mean out-of-sample capacity")
    axes[0].set_title("Cross-time product capacity u[a]*u[b]\n(additive/static models are 0 by construction)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    cls = df.groupby(["variant", "target_class"])["capacity"].mean().unstack()
    cls = cls[["linear", "same_time_nl", "cross_time"]]
    cls.plot.bar(ax=axes[1], rot=30)
    axes[1].set_ylabel("mean capacity")
    axes[1].set_title("Capacity by target class")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png = args.out_dir / f"temporal_memory_diagnostic_{args.tag}.png"
    fig.savefig(out_png, dpi=150)

    print("\n=== Cross-time capacity by separation ===")
    print(pivot.round(3).to_string())
    print("\n=== Mean capacity by target class ===")
    print(cls.round(3).to_string())
    print(f"\nWrote {out_csv}\nWrote {out_json}\nWrote {out_png}")


if __name__ == "__main__":
    main()
