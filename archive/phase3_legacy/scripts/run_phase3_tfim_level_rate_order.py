#!/usr/bin/env python3
"""TFIM anchor-order test on the same two level/rate channels as Rydberg."""
from __future__ import annotations

import argparse, json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import scale_splits_train_only
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.qrc.rydberg_reservoir import ensure_market_scalar
from qpitome_qrc.qrc.tfim_reservoir import (
    TFIMQRCConfig, fit_qrc_readout, fit_tfim_qrc_regressor,
    make_qrc_sequence_splits, predict_qrc_readout, select_anchor_indices,
    summarize_qrc_result,
)

TARGET = "future_rv_20d"
FEATURES = ["vix_rv_spread", "rv_accel_log_5_20"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--anchor-policy", choices=("even", "recent"), default="even")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--observable-mode", choices=("z", "zx", "zxzz"), default="z")
    p.add_argument("--collect-anchor-features", action="store_true")
    p.add_argument("--evolution-time", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("scratch/tfim_anchor_order"))
    p.add_argument("--tag", default="level_rate")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def raw_features(X, anchor_idx):
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]))
    return np.column_stack([anchors] + stats)


def warn(H, y, q):
    thr = float(np.quantile(y["train"], q))
    lab = {k: (v >= thr).astype(int) for k, v in y.items()}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    clf.fit(H["train"], lab["train"])
    out = {"threshold": thr}
    for split in ("val", "test"):
        score = clf.predict_proba(H[split])[:, 1]
        pred = (score >= 0.5).astype(int)
        yl = lab[split]
        out[split] = dict(
            n=int(len(yl)), n_pos=int(yl.sum()), auc=float(roc_auc_score(yl, score)),
            ap=float(average_precision_score(yl, score)), f1=float(f1_score(yl, pred, zero_division=0)),
            precision=float(precision_score(yl, pred, zero_division=0)),
            recall=float(recall_score(yl, pred, zero_division=0)),
        )
    return out


def main():
    a = parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    for col in FEATURES:
        df = ensure_market_scalar(df, col)
    splits, _ = scale_splits_train_only(chronological_tabular_split(df), feature_columns=FEATURES)
    seq = make_qrc_sequence_splits(splits, feature_columns=FEATURES, target_column=TARGET, lookback_days=a.lookback)
    if a.stride > 1:
        seq = {k: (X[::a.stride], y[::a.stride], d[::a.stride].reset_index(drop=True)) for k, (X, y, d) in seq.items()}
    seq_rev = {k: (X[:, ::-1, :], y, d) for k, (X, y, d) in seq.items()}
    y = {k: v[1] for k, v in seq.items()}
    print({k: len(v) for k, v in y.items()})

    cfg = TFIMQRCConfig(
        qubits=a.qubits, pca_components=2, lookback_days=a.lookback, anchor_count=a.anchors,
        anchor_policy=a.anchor_policy, observable_mode=a.observable_mode,
        collect_anchor_features=a.collect_anchor_features, evolution_time=a.evolution_time,
    )
    variants = {"raw_baseline": (None, seq), "tfim_chronological": (cfg, seq), "tfim_reverse": (cfg, seq_rev), "tfim_recent": (replace(cfg, anchor_policy="recent"), seq)}
    anchor_idx = select_anchor_indices(a.lookback, a.anchors, a.anchor_policy)
    summary = {"tag": a.tag, "target": TARGET, "features": FEATURES, "variants": {}}
    rows = []
    for name, (config, s) in variants.items():
        print(f"\n=== {name} ===")
        if config is None:
            H = {k: raw_features(s[k][0], anchor_idx) for k in s}
            row = {"model": name, "n_reservoir_features": H["train"].shape[1]}
            readout, scaler = fit_qrc_readout(H["train"], y["train"], config=cfg)
            for split in ("train", "val", "test"):
                m = evaluate_volatility_forecast(y[split], predict_qrc_readout(readout, scaler, H[split], config=cfg))
                row[f"{split}_rmse"], row[f"{split}_qlike"], row[f"{split}_mz_r2"] = m.rmse, m.qlike, m.mz_r2
        else:
            res = fit_tfim_qrc_regressor(s, config=config, target=TARGET, verbose=a.verbose)
            row = summarize_qrc_result(res); row["model"] = name
            H = {"train": res.train_features, "val": res.val_features, "test": res.test_features}
        summary["variants"][name] = {"regression": {k: row[k] for k in row if k.startswith(("train_", "val_", "test_"))}, "q90_warning": warn(H, y, 0.90), "q95_crisis": warn(H, y, 0.95)}
        rows.append(row)
        for task in ("q90_warning", "q95_crisis"):
            t = summary["variants"][name][task]["test"]
            print(f"{task} test: AUC={t['auc']:.3f} AP={t['ap']:.3f} F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f}")
    outj = a.out_dir / f"tfim_level_rate_order_{a.tag}.json"
    outc = a.out_dir / f"tfim_level_rate_order_{a.tag}.csv"
    outj.write_text(json.dumps(summary, indent=2, default=str)); pd.DataFrame(rows).to_csv(outc, index=False)
    print(f"\nWrote {outj}\nWrote {outc}")

if __name__ == "__main__":
    main()
