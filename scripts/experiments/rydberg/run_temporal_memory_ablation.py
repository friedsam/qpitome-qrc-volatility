#!/usr/bin/env python3
"""Ablate temporal memory in the frozen two-channel Rydberg reservoir."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.targets import (
    RV_INNOVATION_TARGET,
    RV_LEVEL_TARGET,
    RV_REFERENCE_COLUMN,
    add_rv_innovation_target,
    reconstruct_future_rv,
)
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.evaluation.transition import evaluate_transition_forecast
from qpitome_qrc.evaluation.walkforward import make_purged_walkforward_folds
from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    build_rydberg_feature_matrix,
    make_level_rate_sequence_splits,
)

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
OUT = Path("results/experiments/rydberg_temporal_memory_ablation_v1")
BASE = RydbergQRCConfig(
    geometry="dual_chain",
    lookback_days=40,
    anchor_count=8,
    total_time_us=0.55,
    reverse_anchors=True,
    encoding="plateau",
    memory_mode="temporal",
    observable_mode="n_nn",
    shots=None,
)
VARIANTS = {
    "temporal": BASE,
    "memoryless": replace(BASE, memory_mode="memoryless"),
    "shuffled": replace(BASE, shuffle_anchors=True, shuffle_seed=1234),
}


def fit_val_selected(f_tr, y_tr, f_va, y_va, f_te):
    scaler = StandardScaler().fit(f_tr)
    tr, va, te = (scaler.transform(x) for x in (f_tr, f_va, f_te))
    best_rmse, best_alpha = np.inf, None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(tr, y_tr)
        rmse = float(np.sqrt(np.mean((model.predict(va) - y_va) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, alpha
    model = Ridge(alpha=best_alpha).fit(tr, y_tr)
    return model.predict(te), best_alpha, best_rmse


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = add_rv_innovation_target(
        pd.read_csv("data/processed/phase2_spy_vix_volatility.csv")
        .sort_values("date")
        .reset_index(drop=True)
    )
    folds = make_purged_walkforward_folds(
        len(frame), n_folds=5, min_train=2500, val_size=504, purge=60
    )
    required = list(dict.fromkeys(
        FEATURE_COLUMNS + [RV_INNOVATION_TARGET, RV_LEVEL_TARGET, RV_REFERENCE_COLUMN, "date"]
    ))
    numeric = [c for c in required if c != "date"]
    rows = []

    for fold in folds:
        fid = int(fold["fold"])
        splits = {}
        for name in ("train", "val", "test"):
            df = frame.iloc[slice(*fold[name])].reset_index(drop=True).copy()
            df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
            splits[name] = df.dropna(subset=required).reset_index(drop=True)

        seq = make_level_rate_sequence_splits(
            splits,
            level_col="vix_rv_spread",
            rate_col="rv_accel_log_5_20",
            target_column=RV_INNOVATION_TARGET,
            lookback_days=40,
        )
        y = {name: np.asarray(values[1], float) for name, values in seq.items()}
        dates = {name: np.asarray(values[2]) for name, values in seq.items()}
        aligned = {
            name: splits[name].set_index("date", drop=False).loc[list(dates[name])].reset_index(drop=True)
            for name in seq
        }

        for label, config in VARIANTS.items():
            feats = {
                name: build_rydberg_feature_matrix(values[0], config)
                for name, values in seq.items()
            }
            pred, alpha, val_rmse = fit_val_selected(
                feats["train"], y["train"], feats["val"], y["val"], feats["test"]
            )
            innovation = evaluate_transition_forecast(y["test"], pred)
            reference = aligned["test"][RV_REFERENCE_COLUMN].to_numpy(float)
            true_future = aligned["test"][RV_LEVEL_TARGET].to_numpy(float)
            pred_future = reconstruct_future_rv(reference, pred)
            competition = asdict(evaluate_volatility_forecast(true_future, pred_future))
            rows.append({
                "fold": fid,
                "variant": label,
                "selected_alpha": alpha,
                "val_innovation_rmse": val_rmse,
                **innovation,
                **{f"reconstructed_{k}": v for k, v in competition.items()},
            })
            print(
                f"fold {fid} {label:10s} "
                f"R2={innovation['innovation_r2']:+.4f} "
                f"RMSE={competition['rmse']:.4f} "
                f"QLIKE={competition['qlike']:.4f} "
                f"MZ_beta={competition['mz_beta']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_fold_metrics.csv", index=False)
    cols = [
        "innovation_rmse", "innovation_r2", "reconstructed_rmse",
        "reconstructed_qlike", "reconstructed_mz_alpha",
        "reconstructed_mz_beta", "reconstructed_mz_r2",
    ]
    summary = df.groupby("variant")[cols].median().sort_values("innovation_r2", ascending=False)
    summary.to_csv(OUT / "median_metrics.csv")
    print("\nMedian metrics:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
