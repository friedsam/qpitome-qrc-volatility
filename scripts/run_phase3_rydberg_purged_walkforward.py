#!/usr/bin/env python3
"""Purged walk-forward validation for the frozen Rydberg scalar reservoir.

This script evaluates the frozen two-channel Rydberg setup across expanding
chronological folds. Each fold uses:

  train | validation | purge gap | test

The purge gap reduces leakage from overlapping future_rv_20d labels and rolling
lookback windows. Outputs stay in scratch/ by default.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    fit_rydberg_qrc_regressor,
    make_level_rate_sequence_splits,
    validate_aquila_feasibility,
)
from qpitome_qrc.qrc.tfim_reservoir import fit_qrc_readout, predict_qrc_readout, select_anchor_indices

TARGET = "future_rv_20d"
SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--level-col", default="vix_rv_spread")
    p.add_argument("--rate-col", default="rv_accel_log_5_20")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
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
    p.add_argument("--stride", type=int, default=1, help="Optional within-fold subsampling for smoke tests")
    p.add_argument("--variants", nargs="*", default=["raw_baseline", "rydberg_temporal", "rydberg_memoryless", "rydberg_shuffled", "rydberg_constant_omega"],
                   help="Subset: raw_baseline raw_products rydberg_temporal rydberg_memoryless rydberg_shuffled rydberg_constant_omega rydberg_omega_zero rydberg_ramp rydberg_ramp_memoryless rydberg_ramp_shuffled")
    p.add_argument("--out-dir", type=Path, default=Path("scratch/rydberg_walkforward"))
    p.add_argument("--tag", default="tt055_a8_reverse")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def make_folds(n: int, *, n_folds: int, min_train: int, val_size: int, purge: int) -> list[dict]:
    first_test_start = min_train + val_size + purge
    if first_test_start >= n:
        raise ValueError("Not enough rows for requested min_train/val_size/purge")
    test_size = (n - first_test_start) // n_folds
    if test_size < 100:
        raise ValueError(f"test_size too small: {test_size}")
    folds = []
    for i in range(n_folds):
        test_start = first_test_start + i * test_size
        test_end = n if i == n_folds - 1 else test_start + test_size
        val_end = test_start - purge
        val_start = val_end - val_size
        train_end = val_start
        if train_end < min_train:
            raise ValueError("Invalid fold construction")
        folds.append(dict(fold=i + 1, train=(0, train_end), val=(val_start, val_end), purge=(val_end, test_start), test=(test_start, test_end)))
    return folds


def date_range(df: pd.DataFrame, sl: tuple[int, int]) -> tuple[str, str]:
    a, b = sl
    return str(df["date"].iloc[a]), str(df["date"].iloc[b - 1])


def raw_anchor_indices(args) -> np.ndarray:
    idx = select_anchor_indices(args.lookback, args.anchors, args.anchor_policy)
    if args.reverse_anchors:
        idx = idx[::-1]
    return np.asarray(idx, dtype=int)


def raw_features(X: np.ndarray, anchor_idx: np.ndarray, *, products: bool = False) -> np.ndarray:
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]))
    feats = [anchors] + stats
    if products:
        # Product-augmented classical baseline: explicit pairwise products of
        # all anchor values (both channels). If the temporal reservoir only
        # MATCHES this, its memory is effectively second-order and classically
        # replicable; if it EXCEEDS it, higher-order many-body memory is doing
        # work on the task.
        m = anchors.shape[1]
        prods = np.stack(
            [anchors[:, a] * anchors[:, b] for a in range(m - 1) for b in range(a + 1, m)],
            axis=1,
        )
        feats.append(prods)
    return np.column_stack(feats)


def warning_head(H: dict[str, np.ndarray], y: dict[str, np.ndarray], quantile: float) -> dict:
    thr = float(np.quantile(y["train"], quantile))
    lab = {k: (v >= thr).astype(int) for k, v in y.items()}
    out = {"threshold": thr, "train_positive_rate": float(lab["train"].mean())}
    if lab["train"].sum() == 0 or lab["train"].sum() == len(lab["train"]):
        return {**out, "note": "degenerate train labels"}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    clf.fit(H["train"], lab["train"])
    for split in ("val", "test"):
        yl = lab[split]
        if yl.sum() == 0 or yl.sum() == len(yl):
            out[split] = {"n": int(len(yl)), "n_pos": int(yl.sum()), "note": "degenerate labels"}
            continue
        score = clf.predict_proba(H[split])[:, 1]
        pred = (score >= 0.5).astype(int)
        out[split] = dict(n=int(len(yl)), n_pos=int(yl.sum()), auc=float(roc_auc_score(yl, score)),
                          ap=float(average_precision_score(yl, score)), f1=float(f1_score(yl, pred, zero_division=0)),
                          precision=float(precision_score(yl, pred, zero_division=0)),
                          recall=float(recall_score(yl, pred, zero_division=0)))
    return out


def metric_row(fold_id: int, model: str, reg: dict, q90: dict, q95: dict, meta: dict) -> dict:
    row = {"fold": fold_id, "model": model, **meta}
    for k in ("test_rmse", "test_qlike", "test_mz_r2"):
        row[k] = reg.get(k)
    for label, block in (("q90", q90), ("q95", q95)):
        test = block.get("test", {})
        for k in ("auc", "ap", "f1", "precision", "recall", "n_pos"):
            row[f"{label}_{k}"] = test.get(k)
    return row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    folds = make_folds(len(df), n_folds=args.n_folds, min_train=args.min_train, val_size=args.val_size, purge=args.purge)

    base = RydbergQRCConfig(
        n_atoms_slow=args.n_slow, n_atoms_fast=args.n_fast,
        spacing_slow_um=args.spacing_slow_um, spacing_fast_um=args.spacing_fast_um,
        lookback_days=args.lookback, anchor_count=args.anchors, anchor_policy=args.anchor_policy,
        reverse_anchors=args.reverse_anchors, shuffle_seed=args.shuffle_seed,
        total_time_us=args.total_time_us, delta_center_rad_us=args.delta_center,
        delta_span_rad_us=args.delta_span, omega_base_rad_us=args.omega_base,
        omega_mod_frac=args.omega_mod_frac, shots=args.shots,
    )
    configs = {
        "rydberg_temporal": base,
        "rydberg_memoryless": replace(base, memory_mode="memoryless"),
        "rydberg_shuffled": replace(base, shuffle_anchors=True),
        "rydberg_constant_omega": replace(base, omega_mode="constant"),
        "rydberg_omega_zero": replace(base, omega_base_rad_us=0.0, omega_mod_frac=0.0),
        # Landau-Zener ramp encoding: level -> Delta value, rate -> Delta slope
        # (native diabatic rate-sensing). Omega fixed as pure mixing drive.
        "rydberg_ramp": replace(base, encoding="ramp", omega_mode="constant"),
        "rydberg_ramp_memoryless": replace(base, encoding="ramp", omega_mode="constant", memory_mode="memoryless"),
        "rydberg_ramp_shuffled": replace(base, encoding="ramp", omega_mode="constant", shuffle_anchors=True),
    }
    feasibility = validate_aquila_feasibility(base)
    print("Aquila feasibility:", feasibility["feasible"], feasibility["checks"])

    rows = []
    summary = {"tag": args.tag, "config": {k: getattr(args, k) for k in vars(args) if k not in {"data", "out_dir"}},
               "feasibility": feasibility, "folds": []}
    anchor_idx = raw_anchor_indices(args)

    for f in folds:
        fold_id = f["fold"]
        split_frames = {
            name: df.iloc[f[name][0]:f[name][1]].copy().reset_index(drop=True)
            for name in SPLIT_NAMES
        }
        seq = make_level_rate_sequence_splits(split_frames, level_col=args.level_col, rate_col=args.rate_col, target_column=TARGET, lookback_days=args.lookback)
        if args.stride > 1:
            seq = {k: (X[::args.stride], y[::args.stride], d[::args.stride].reset_index(drop=True)) for k, (X, y, d) in seq.items()}
        y = {k: v[1] for k, v in seq.items()}
        meta = {"train_dates": date_range(df, f["train"]), "val_dates": date_range(df, f["val"]),
                "purge_dates": date_range(df, f["purge"]), "test_dates": date_range(df, f["test"]),
                "train_n": int(len(y["train"])), "val_n": int(len(y["val"])), "test_n": int(len(y["test"]))}
        print(f"\n=== Fold {fold_id}: test {meta['test_dates']} ===")
        fold_summary = {"fold": fold_id, **meta, "variants": {}}

        for model in args.variants:
            print(f"-- {model}")
            if model in ("raw_baseline", "raw_products"):
                products = model == "raw_products"
                H = {k: raw_features(seq[k][0], anchor_idx, products=products) for k in seq}
                readout, scaler = fit_qrc_readout(H["train"], y["train"], config=base)
                reg = {"model": model}
                for split in ("train", "val", "test"):
                    m = evaluate_volatility_forecast(y[split], predict_qrc_readout(readout, scaler, H[split], config=base))
                    reg[f"{split}_rmse"], reg[f"{split}_qlike"], reg[f"{split}_mz_r2"] = m.rmse, m.qlike, m.mz_r2
            else:
                if model not in configs:
                    raise ValueError(f"Unknown variant: {model}")
                result = fit_rydberg_qrc_regressor(seq, config=configs[model], target=TARGET, verbose=args.verbose)
                H = {"train": result.train_features, "val": result.val_features, "test": result.test_features}
                reg = {"model": model, "test_rmse": result.test_metrics.rmse,
                       "test_qlike": result.test_metrics.qlike, "test_mz_r2": result.test_metrics.mz_r2}
                reg.update({"train_rmse": result.train_metrics.rmse, "val_rmse": result.val_metrics.rmse})
            q90 = warning_head(H, y, 0.90); q95 = warning_head(H, y, 0.95)
            fold_summary["variants"][model] = {"regression": reg, "q90_warning": q90, "q95_crisis": q95}
            rows.append(metric_row(fold_id, model, reg, q90, q95, meta))
            t = q95.get("test", {})
            if "auc" in t:
                print(f"   q95 AUC={t['auc']:.3f} AP={t['ap']:.3f} F1={t['f1']:.3f}")
            else:
                print(f"   q95 skipped: {t.get('note', q95.get('note', 'unknown'))}")
        summary["folds"].append(fold_summary)

    wide = pd.DataFrame(rows)
    out_csv = args.out_dir / f"rydberg_walkforward_{args.tag}.csv"
    out_json = args.out_dir / f"rydberg_walkforward_{args.tag}.json"
    wide.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Median test metrics by model ===")
    cols = ["test_rmse", "test_qlike", "test_mz_r2", "q95_auc", "q95_ap", "q95_f1"]
    print(wide.groupby("model")[cols].median(numeric_only=True).to_string())
    print(f"\nWrote {out_csv}\nWrote {out_json}")


if __name__ == "__main__":
    main()
