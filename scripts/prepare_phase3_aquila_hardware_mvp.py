#!/usr/bin/env python3
"""Prepare Aquila hardware-transfer windows for the Phase 3 Rydberg reservoir.

This script does not submit jobs. It creates a frozen, reproducible hardware
run package that can be used when Aquila is online.

Outputs:
  scratch/aquila_mvp/
    fold5_per_window_predictions.csv
    selected_hardware_windows_validation2.csv
    selected_hardware_windows_validation4.csv
    selected_windows_summary.json
    aquila_run_specs_validation2.json
    aquila_run_specs_validation4.json

The intended immediate hardware use is validation2 only:
  - one q95-positive high-score window
  - one quiet low-score negative window
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.walkforward import (
    make_purged_walkforward_folds,
    slice_fold_frames,
)
from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    fit_rydberg_qrc_regressor,
    make_level_rate_sequence_splits,
    validate_aquila_feasibility,
)
from qpitome_qrc.qrc.tfim_reservoir import (
    fit_qrc_readout,
    predict_qrc_readout,
    select_anchor_indices,
)

TARGET = "future_rv_20d"
SPLIT_NAMES = ("train", "val", "test")


def make_folds(n: int, *, n_folds: int, min_train: int, val_size: int, purge: int) -> list[dict]:
    """Backward-compatible wrapper around the shared Phase 3 protocol."""
    return make_purged_walkforward_folds(
        n,
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
    )


def raw_features(X: np.ndarray, anchor_idx: np.ndarray, *, products: bool = False) -> np.ndarray:
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]))

    feats = [anchors] + stats

    if products:
        m = anchors.shape[1]
        prods = np.stack(
            [anchors[:, a] * anchors[:, b] for a in range(m - 1) for b in range(a + 1, m)],
            axis=1,
        )
        feats.append(prods)

    return np.column_stack(feats)


def fit_warning_scores(H: dict[str, np.ndarray], y: dict[str, np.ndarray], quantile: float):
    thr = float(np.quantile(y["train"], quantile))
    labels = {k: (v >= thr).astype(int) for k, v in y.items()}

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
    )
    clf.fit(H["train"], labels["train"])

    scores = {k: clf.predict_proba(H[k])[:, 1] for k in H}
    return thr, labels, scores


def build_config(args: argparse.Namespace) -> RydbergQRCConfig:
    return RydbergQRCConfig(
        n_atoms_slow=args.n_slow,
        n_atoms_fast=args.n_fast,
        spacing_slow_um=args.spacing_slow_um,
        spacing_fast_um=args.spacing_fast_um,
        row_gap_um=args.row_gap_um,
        lookback_days=args.lookback,
        anchor_count=args.anchors,
        anchor_policy=args.anchor_policy,
        reverse_anchors=args.reverse_anchors,
        shuffle_seed=args.shuffle_seed,
        total_time_us=args.total_time_us,
        delta_center_rad_us=args.delta_center,
        delta_span_rad_us=args.delta_span,
        omega_base_rad_us=args.omega_base,
        omega_mod_frac=args.omega_mod_frac,
        encoding=args.encoding,
        shots=args.shots,
    )


def make_run_specs(selected: pd.DataFrame, config: RydbergQRCConfig) -> list[dict]:
    """Create provider-neutral run specs.

    These are not Braket SDK objects. They are stable JSON specs containing the
    frozen market-window inputs and Rydberg candidate configuration. The actual
    provider submission wrapper can consume these specs when Aquila is online.
    """
    specs = []
    for _, row in selected.iterrows():
        specs.append(
            {
                "date": row["date"],
                "selection_bucket": row["selection_bucket"],
                "q90_label": int(row["q90_label"]),
                "q95_label": int(row["q95_label"]),
                "future_rv_20d": float(row["future_rv_20d"]),
                "simulator_scores": {
                    "rydberg_temporal_q90_score": float(row["rydberg_temporal_q90_score"]),
                    "rydberg_temporal_q95_score": float(row["rydberg_temporal_q95_score"]),
                    "raw_baseline_q90_score": float(row["raw_baseline_q90_score"]),
                    "raw_baseline_q95_score": float(row["raw_baseline_q95_score"]),
                },
                "hardware_candidate": {
                    "geometry": "dual_chain",
                    "n_atoms_slow": config.n_atoms_slow,
                    "n_atoms_fast": config.n_atoms_fast,
                    "spacing_slow_um": config.spacing_slow_um,
                    "spacing_fast_um": config.spacing_fast_um,
                    "row_gap_um": config.row_gap_um,
                    "lookback_days": config.lookback_days,
                    "anchor_count": config.anchor_count,
                    "anchor_policy": config.anchor_policy,
                    "reverse_anchors": config.reverse_anchors,
                    "total_time_us": config.total_time_us,
                    "delta_center_rad_us": config.delta_center_rad_us,
                    "delta_span_rad_us": config.delta_span_rad_us,
                    "omega_base_rad_us": config.omega_base_rad_us,
                    "omega_mod_frac": config.omega_mod_frac,
                    "encoding": config.encoding,
                    "shots": config.shots,
                    "observables": "occupations + all pair correlations",
                },
                "intended_use": "hardware-transfer sanity check, not benchmark",
            }
        )
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("scratch/aquila_mvp"))

    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)

    p.add_argument("--level-col", default="vix_rv_spread")
    p.add_argument("--rate-col", default="rv_accel_log_5_20")

    p.add_argument("--n-slow", type=int, default=4)
    p.add_argument("--n-fast", type=int, default=4)
    p.add_argument("--spacing-slow-um", type=float, default=9.0)
    p.add_argument("--spacing-fast-um", type=float, default=15.0)
    p.add_argument("--row-gap-um", type=float, default=12.0)

    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--anchor-policy", choices=("even", "recent"), default="even")
    p.add_argument("--reverse-anchors", action="store_true", default=True)
    p.add_argument("--shuffle-seed", type=int, default=1234)

    p.add_argument("--total-time-us", type=float, default=0.55)
    p.add_argument("--delta-center", type=float, default=6.0)
    p.add_argument("--delta-span", type=float, default=4.0)
    p.add_argument("--omega-base", type=float, default=6.0)
    p.add_argument("--omega-mod-frac", type=float, default=0.5)
    p.add_argument("--encoding", choices=("plateau", "ramp"), default="plateau")
    p.add_argument("--shots", type=int, default=1000)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    feasibility = validate_aquila_feasibility(config)

    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    folds = make_folds(
        len(df),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    fold = folds[-1]

    split_frames = slice_fold_frames(df, fold)

    seq = make_level_rate_sequence_splits(
        split_frames,
        level_col=args.level_col,
        rate_col=args.rate_col,
        target_column=TARGET,
        lookback_days=args.lookback,
    )

    y = {k: seq[k][1] for k in seq}
    dates = {k: pd.to_datetime(seq[k][2]) for k in seq}

    anchor_idx = select_anchor_indices(args.lookback, args.anchors, args.anchor_policy)
    if args.reverse_anchors:
        anchor_idx = anchor_idx[::-1]

    models = {}

    H_raw = {k: raw_features(seq[k][0], anchor_idx, products=False) for k in seq}
    raw_readout, raw_scaler = fit_qrc_readout(H_raw["train"], y["train"], config=config)
    raw_pred = {
        k: predict_qrc_readout(raw_readout, raw_scaler, H_raw[k], config=config)
        for k in seq
    }
    models["raw_baseline"] = {"H": H_raw, "pred": raw_pred}

    variant_configs = {
        "rydberg_temporal": config,
        "rydberg_memoryless": replace(config, memory_mode="memoryless"),
        "rydberg_shuffled": replace(config, shuffle_anchors=True),
    }

    for name, cfg in variant_configs.items():
        result = fit_rydberg_qrc_regressor(seq, config=cfg, target=TARGET, verbose=False)
        models[name] = {
            "H": {
                "train": result.train_features,
                "val": result.val_features,
                "test": result.test_features,
            },
            "pred": {
                "train": result.train_predictions,
                "val": result.val_predictions,
                "test": result.test_predictions,
            },
        }

    q90_thr = float(np.quantile(y["train"], 0.90))
    q95_thr = float(np.quantile(y["train"], 0.95))

    base = pd.DataFrame(
        {
            "date": dates["test"].dt.strftime("%Y-%m-%d"),
            "future_rv_20d": y["test"],
            "q90_label": (y["test"] >= q90_thr).astype(int),
            "q95_label": (y["test"] >= q95_thr).astype(int),
        }
    )

    for model_name, pack in models.items():
        H = pack["H"]
        pred = pack["pred"]

        base[f"{model_name}_rv_pred"] = pred["test"]

        _, _, q90_scores = fit_warning_scores(H, y, 0.90)
        _, _, q95_scores = fit_warning_scores(H, y, 0.95)

        base[f"{model_name}_q90_score"] = q90_scores["test"]
        base[f"{model_name}_q95_score"] = q95_scores["test"]

    dfp = base.copy()

    q95_pos = dfp[dfp["q95_label"] == 1].sort_values(
        "rydberg_temporal_q95_score", ascending=False
    )
    q90_only = dfp[(dfp["q90_label"] == 1) & (dfp["q95_label"] == 0)].sort_values(
        "rydberg_temporal_q90_score", ascending=False
    )
    quiet = dfp[(dfp["q90_label"] == 0) & (dfp["q95_label"] == 0)].sort_values(
        "rydberg_temporal_q90_score", ascending=True
    )
    hard_neg = dfp[(dfp["q90_label"] == 0) & (dfp["q95_label"] == 0)].sort_values(
        "rydberg_temporal_q95_score", ascending=False
    )

    selected_validation2 = pd.concat(
        [
            q95_pos.head(1).assign(selection_bucket="q95_positive_high_score"),
            quiet.head(1).assign(selection_bucket="quiet_low_score"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["date"]).reset_index(drop=True)

    selected_validation4 = pd.concat(
        [
            q95_pos.head(1).assign(selection_bucket="q95_positive_high_score"),
            q90_only.head(1).assign(selection_bucket="q90_only_high_score"),
            quiet.head(1).assign(selection_bucket="quiet_low_score"),
            hard_neg.head(1).assign(selection_bucket="hard_negative_high_score"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["date"]).reset_index(drop=True)

    base.to_csv(args.out_dir / "fold5_per_window_predictions.csv", index=False)
    selected_validation2.to_csv(args.out_dir / "selected_hardware_windows_validation2.csv", index=False)
    selected_validation4.to_csv(args.out_dir / "selected_hardware_windows_validation4.csv", index=False)

    specs2 = make_run_specs(selected_validation2, config)
    specs4 = make_run_specs(selected_validation4, config)

    (args.out_dir / "aquila_run_specs_validation2.json").write_text(json.dumps(specs2, indent=2))
    (args.out_dir / "aquila_run_specs_validation4.json").write_text(json.dumps(specs4, indent=2))

    summary = {
        "purpose": "Aquila transfer sanity-check package; not a benchmark",
        "fold": 5,
        "test_dates": [
            str(df["date"].iloc[fold["test"][0]].date()),
            str(df["date"].iloc[fold["test"][1] - 1].date()),
        ],
        "candidate": {
            "total_time_us": config.total_time_us,
            "row_gap_um": config.row_gap_um,
            "shots": config.shots,
            "encoding": config.encoding,
            "geometry": "dual_chain",
            "n_atoms": config.n_atoms_slow + config.n_atoms_fast,
        },
        "feasibility": feasibility,
        "thresholds": {
            "q90_future_rv_20d": q90_thr,
            "q95_future_rv_20d": q95_thr,
        },
        "validation2_count": int(len(selected_validation2)),
        "validation4_count": int(len(selected_validation4)),
        "estimated_cost_usd": {
            "per_window_1000_shots": 10.30,
            "validation2": round(10.30 * len(selected_validation2), 2),
            "validation4": round(10.30 * len(selected_validation4), 2),
        },
    }

    (args.out_dir / "selected_windows_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
