#!/usr/bin/env python3
"""TFIM-QRC anchor-order comparison for the Phase 3 Rydberg result.

This runner answers one fairness question raised by the reverse-order Rydberg
candidate: does reversing the temporal injection order also improve the Phase 2
TFIM reservoir?

It evaluates identical train/val/test windows under:

  raw_baseline        anchor values + window summaries, no quantum dynamics
  tfim_chronological  normal oldest-to-newest TFIM injection
  tfim_reverse        same selected anchors, injected newest-to-oldest by
                      reversing each rolling window before TFIM evolution
  tfim_recent         optional recent-weighted anchor selection

No future information is introduced by reversal: every model still sees only the
trailing lookback window ending at the forecast date.

Outputs go to scratch/ by default (git-ignored).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import DEFAULT_FEATURE_COLUMNS, scale_splits_train_only
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.qrc.tfim_reservoir import (
    TFIMQRCConfig,
    fit_qrc_readout,
    fit_tfim_qrc_regressor,
    make_qrc_sequence_splits,
    predict_qrc_readout,
    select_anchor_indices,
    summarize_qrc_result,
)

TARGET = "future_rv_20d"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--target", default=TARGET)
    p.add_argument("--feature-cols", nargs="*", default=None, help="Defaults to qpitome_qrc.data.features.DEFAULT_FEATURE_COLUMNS")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--anchor-policy", choices=("even", "recent"), default="even")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--observable-mode", choices=("z", "zx", "zxzz"), default="z")
    p.add_argument("--trotter-steps-per-anchor", type=int, default=1)
    p.add_argument("--virtual-nodes-per-anchor", type=int, default=1)
    p.add_argument("--collect-anchor-features", action="store_true")
    p.add_argument("--topology", choices=("chain", "full"), default="chain")
    p.add_argument("--coupling-scale", type=float, default=0.7)
    p.add_argument("--transverse-field", type=float, default=0.5)
    p.add_argument("--evolution-time", type=float, default=0.5)
    p.add_argument("--angle-max", type=float, default=float(np.pi / 2))
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--scaler", choices=("standard", "robust", "none"), default="standard")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("scratch/tfim_anchor_order"))
    p.add_argument("--tag", default="v1")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def reverse_sequence_splits(
    seq: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]],
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.Series]]:
    """Reverse each trailing window in time while preserving y and dates."""
    return {name: (X[:, ::-1, :], y, d) for name, (X, y, d) in seq.items()}


def raw_baseline_features(X: np.ndarray, anchor_indices: np.ndarray) -> np.ndarray:
    """Anchor values of all channels + per-channel window summaries."""
    anchors = X[:, anchor_indices, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(
            np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)])
        )
    return np.column_stack([anchors] + stats)


def warning_head(
    H: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    quantile: float,
    seed: int = 42,
) -> dict:
    """Train-quantile warning labels + logistic readout."""
    threshold = float(np.quantile(y["train"], quantile))
    labels = {k: (v >= threshold).astype(int) for k, v in y.items()}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )
    clf.fit(H["train"], labels["train"])

    out: dict = {"threshold": threshold, "train_positive_rate": float(labels["train"].mean())}
    for split in ("val", "test"):
        yl = labels[split]
        if yl.sum() == 0 or yl.sum() == len(yl):
            out[split] = {"note": "degenerate labels", "n_pos": int(yl.sum())}
            continue
        score = clf.predict_proba(H[split])[:, 1]
        pred = (score >= 0.5).astype(int)
        out[split] = {
            "n": int(len(yl)),
            "n_pos": int(yl.sum()),
            "auc": float(roc_auc_score(yl, score)),
            "ap": float(average_precision_score(yl, score)),
            "f1": float(f1_score(yl, pred, zero_division=0)),
            "precision": float(precision_score(yl, pred, zero_division=0)),
            "recall": float(recall_score(yl, pred, zero_division=0)),
        }
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = args.feature_cols or list(DEFAULT_FEATURE_COLUMNS)
    df = pd.read_csv(args.data)
    splits_raw = chronological_tabular_split(df)
    splits, _ = scale_splits_train_only(
        splits_raw,
        feature_columns=feature_cols,
        scaler_name=None if args.scaler == "none" else args.scaler,
    )
    seq = make_qrc_sequence_splits(
        splits,
        feature_columns=feature_cols,
        target_column=args.target,
        lookback_days=args.lookback,
    )
    if args.stride > 1:
        seq = {
            k: (X[:: args.stride], y[:: args.stride], d[:: args.stride].reset_index(drop=True))
            for k, (X, y, d) in seq.items()
        }
    seq_reverse = reverse_sequence_splits(seq)
    y = {k: v[1] for k, v in seq.items()}
    print({k: len(v) for k, v in y.items()})

    base_config = TFIMQRCConfig(
        qubits=args.qubits,
        pca_components=len(feature_cols),
        lookback_days=args.lookback,
        anchor_count=args.anchors,
        anchor_policy=args.anchor_policy,
        observable_mode=args.observable_mode,
        trotter_steps_per_anchor=args.trotter_steps_per_anchor,
        virtual_nodes_per_anchor=args.virtual_nodes_per_anchor,
        collect_anchor_features=args.collect_anchor_features,
        topology=args.topology,
        coupling_scale=args.coupling_scale,
        transverse_field=args.transverse_field,
        evolution_time=args.evolution_time,
        angle_max=args.angle_max,
        ridge_alpha=args.ridge_alpha,
    )
    recent_config = replace(base_config, anchor_policy="recent")

    variants: dict[str, tuple[TFIMQRCConfig | None, dict[str, tuple[np.ndarray, np.ndarray, pd.Series]]]] = {
        "raw_baseline": (None, seq),
        "tfim_chronological": (base_config, seq),
        "tfim_reverse": (base_config, seq_reverse),
        "tfim_recent": (recent_config, seq),
    }

    anchor_idx = select_anchor_indices(args.lookback, args.anchors, args.anchor_policy)
    summary: dict = {
        "tag": args.tag,
        "target": args.target,
        "feature_cols": feature_cols,
        "stride": args.stride,
        "base_config": asdict(base_config),
        "variants": {},
    }
    rows = []

    for name, (config, variant_seq) in variants.items():
        print(f"\n=== {name} ===")
        if config is None:
            H = {k: raw_baseline_features(variant_seq[k][0], anchor_idx) for k in variant_seq}
            reg_row = {"model": name, "n_reservoir_features": H["train"].shape[1]}
            readout, scaler = fit_qrc_readout(H["train"], y["train"], config=base_config)
            for split in ("train", "val", "test"):
                pred = predict_qrc_readout(readout, scaler, H[split], config=base_config)
                m = evaluate_volatility_forecast(y[split], pred)
                reg_row[f"{split}_rmse"] = m.rmse
                reg_row[f"{split}_qlike"] = m.qlike
                reg_row[f"{split}_mz_r2"] = m.mz_r2
        else:
            result = fit_tfim_qrc_regressor(variant_seq, config=config, target=args.target, verbose=args.verbose)
            reg_row = summarize_qrc_result(result)
            reg_row["model"] = name
            H = {"train": result.train_features, "val": result.val_features, "test": result.test_features}

        variant_summary = {
            "regression": {
                k: reg_row[k]
                for k in reg_row
                if any(k.startswith(s) for s in ("train_", "val_", "test_"))
            },
            "n_features": int(reg_row["n_reservoir_features"]),
            "q90_warning": warning_head(H, y, 0.90),
            "q95_crisis": warning_head(H, y, 0.95),
        }
        summary["variants"][name] = variant_summary
        rows.append(reg_row)

        for task in ("q90_warning", "q95_crisis"):
            t = variant_summary[task].get("test", {})
            if "auc" in t:
                print(
                    f"{task} test: AUC={t['auc']:.3f} AP={t['ap']:.3f} "
                    f"F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f}"
                )

    out_json = args.out_dir / f"tfim_anchor_order_{args.tag}.json"
    out_csv = args.out_dir / f"tfim_anchor_order_{args.tag}.csv"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_json}\nWrote {out_csv}")


if __name__ == "__main__":
    main()
