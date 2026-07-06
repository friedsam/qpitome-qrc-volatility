#!/usr/bin/env python3
"""Rydberg reservoir with full Phase 2 inputs compressed to two PCA channels.

The PCA is fit on the training split only. The two PCA time series are then used
as the same Aquila-native controls as the scalar Rydberg runner:

  PC1(t) -> Delta(t)
  PC2(t) -> Omega(t)

This keeps the two global-control constraint but lets Rydberg see the same raw
feature family as TFIM/ESN through a leakage-safe 2-channel compression.
"""
from __future__ import annotations

import argparse, json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import DEFAULT_FEATURE_COLUMNS, drop_nonfinite_model_rows, scale_splits_train_only
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.qrc.rydberg_reservoir import (
    C6_RAD_UM6_PER_US, RydbergQRCConfig, fit_rydberg_qrc_regressor,
    make_level_rate_sequence_splits, summarize_rydberg_result, validate_aquila_feasibility,
)
from qpitome_qrc.qrc.tfim_reservoir import fit_qrc_readout, predict_qrc_readout, select_anchor_indices

TARGET = "future_rv_20d"
PCA_LEVEL = "pca2_level"
PCA_RATE = "pca2_rate"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--anchor-policy", choices=("even", "recent"), default="even")
    p.add_argument("--reverse-anchors", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=1234)
    p.add_argument("--total-time-us", type=float, default=0.55)
    p.add_argument("--n-slow", type=int, default=4)
    p.add_argument("--n-fast", type=int, default=4)
    p.add_argument("--spacing-slow-um", type=float, default=9.0)
    p.add_argument("--spacing-fast-um", type=float, default=15.0)
    p.add_argument("--delta-center", type=float, default=6.0)
    p.add_argument("--delta-span", type=float, default=4.0)
    p.add_argument("--omega-base", type=float, default=6.0)
    p.add_argument("--omega-mod-frac", type=float, default=0.5)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("scratch/rydberg_temporal_reservoir"))
    p.add_argument("--tag", default="pca2")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def make_pca2_splits(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    splits0 = chronological_tabular_split(df)
    cleaned = {
        k: drop_nonfinite_model_rows(v, feature_columns=list(DEFAULT_FEATURE_COLUMNS), target_columns=[TARGET])
        for k, v in splits0.items()
    }
    scaled, _ = scale_splits_train_only(cleaned, feature_columns=list(DEFAULT_FEATURE_COLUMNS), scaler_name="standard")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaled["train"][DEFAULT_FEATURE_COLUMNS].to_numpy(float))
    out = {}
    for k, split in scaled.items():
        comps = pca.transform(split[DEFAULT_FEATURE_COLUMNS].to_numpy(float))
        frame = split[["date", TARGET]].copy().reset_index(drop=True)
        frame[PCA_LEVEL] = comps[:, 0]
        frame[PCA_RATE] = comps[:, 1]
        out[k] = frame
    info = {"explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]}
    return out, info


def raw_anchor_indices(args):
    idx = select_anchor_indices(args.lookback, args.anchors, args.anchor_policy)
    if args.reverse_anchors:
        idx = idx[::-1]
    return np.asarray(idx, dtype=int)


def raw_features(X, anchor_idx):
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]))
    return np.column_stack([anchors] + stats)


def phase_budget(config):
    k = int(config.anchor_count); t = float(config.total_time_us / k)
    delta_abs = float(abs(config.delta_center_rad_us) + abs(config.delta_span_rad_us))
    omin = float(max(config.omega_base_rad_us * (1 - abs(config.omega_mod_frac)), 0.0))
    omax = float(config.omega_base_rad_us * (1 + abs(config.omega_mod_frac)))
    vslow = float(C6_RAD_UM6_PER_US / config.spacing_slow_um**6)
    vfast = float(C6_RAD_UM6_PER_US / config.spacing_fast_um**6)
    return dict(t_segment_us=t, omega_min_t_segment=omin*t, omega_max_t_segment=omax*t,
                delta_abs_max_t_segment=delta_abs*t, v_slow_nn_t_segment=vslow*t,
                v_fast_nn_t_segment=vfast*t)


def warn(H, y, q):
    thr = float(np.quantile(y["train"], q))
    lab = {k: (v >= thr).astype(int) for k, v in y.items()}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    clf.fit(H["train"], lab["train"])
    out = {"threshold": thr}
    for split in ("val", "test"):
        score = clf.predict_proba(H[split])[:, 1]
        pred = (score >= 0.5).astype(int); yl = lab[split]
        out[split] = dict(n=int(len(yl)), n_pos=int(yl.sum()), auc=float(roc_auc_score(yl, score)),
                          ap=float(average_precision_score(yl, score)), f1=float(f1_score(yl, pred, zero_division=0)),
                          precision=float(precision_score(yl, pred, zero_division=0)),
                          recall=float(recall_score(yl, pred, zero_division=0)))
    return out


def main():
    a = parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    splits, pca_info = make_pca2_splits(pd.read_csv(a.data))
    seq = make_level_rate_sequence_splits(splits, level_col=PCA_LEVEL, rate_col=PCA_RATE, target_column=TARGET, lookback_days=a.lookback)
    if a.stride > 1:
        seq = {k: (X[::a.stride], y[::a.stride], d[::a.stride].reset_index(drop=True)) for k, (X, y, d) in seq.items()}
    y = {k: v[1] for k, v in seq.items()}
    print({k: len(v) for k, v in y.items()})
    print("PCA explained variance ratio:", pca_info["explained_variance_ratio"])

    cfg = RydbergQRCConfig(n_atoms_slow=a.n_slow, n_atoms_fast=a.n_fast,
        spacing_slow_um=a.spacing_slow_um, spacing_fast_um=a.spacing_fast_um,
        lookback_days=a.lookback, anchor_count=a.anchors, anchor_policy=a.anchor_policy,
        reverse_anchors=a.reverse_anchors, shuffle_seed=a.shuffle_seed,
        total_time_us=a.total_time_us, delta_center_rad_us=a.delta_center,
        delta_span_rad_us=a.delta_span, omega_base_rad_us=a.omega_base,
        omega_mod_frac=a.omega_mod_frac, shots=a.shots)
    feas = validate_aquila_feasibility(cfg)
    print("Aquila feasibility:", feas["feasible"], feas["checks"])

    variants = {"raw_baseline": None, "rydberg_temporal": cfg,
        "rydberg_memoryless": replace(cfg, memory_mode="memoryless"),
        "rydberg_shuffled": replace(cfg, shuffle_anchors=True),
        "rydberg_constant_omega": replace(cfg, omega_mode="constant"),
        "rydberg_omega_zero": replace(cfg, omega_base_rad_us=0.0, omega_mod_frac=0.0)}
    anchor_idx = raw_anchor_indices(a)
    summary = {"tag": a.tag, "target": TARGET, "channel_mode": "pca2", "pca": pca_info,
               "feasibility": feas, "base_phase_budget": phase_budget(cfg), "variants": {}}
    rows = []
    for name, config in variants.items():
        print(f"\n=== {name} ===")
        if config is None:
            H = {k: raw_features(seq[k][0], anchor_idx) for k in seq}
            row = {"model": name, "n_reservoir_features": H["train"].shape[1]}
            readout, scaler = fit_qrc_readout(H["train"], y["train"], config=cfg)
            for split in ("train", "val", "test"):
                m = evaluate_volatility_forecast(y[split], predict_qrc_readout(readout, scaler, H[split], config=cfg))
                row[f"{split}_rmse"], row[f"{split}_qlike"], row[f"{split}_mz_r2"] = m.rmse, m.qlike, m.mz_r2
            pb = None
        else:
            res = fit_rydberg_qrc_regressor(seq, config=config, target=TARGET, verbose=a.verbose)
            row = summarize_rydberg_result(res); row["model"] = name
            H = {"train": res.train_features, "val": res.val_features, "test": res.test_features}
            pb = phase_budget(config)
        summary["variants"][name] = {"regression": {k: row[k] for k in row if k.startswith(("train_", "val_", "test_"))},
                                      "n_features": int(row["n_reservoir_features"]), "phase_budget": pb,
                                      "q90_warning": warn(H, y, 0.90), "q95_crisis": warn(H, y, 0.95)}
        rows.append(row)
        for task in ("q90_warning", "q95_crisis"):
            t = summary["variants"][name][task]["test"]
            print(f"{task} test: AUC={t['auc']:.3f} AP={t['ap']:.3f} F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f}")
    outj = a.out_dir / f"rydberg_pca2_reservoir_{a.tag}.json"
    outc = a.out_dir / f"rydberg_pca2_reservoir_{a.tag}.csv"
    outj.write_text(json.dumps(summary, indent=2, default=str)); pd.DataFrame(rows).to_csv(outc, index=False)
    print(f"\nWrote {outj}\nWrote {outc}")

if __name__ == "__main__":
    main()
