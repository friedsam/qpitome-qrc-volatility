#!/usr/bin/env python3
"""Matched-date Rydberg architecture comparison on rv_innovation_20d.

All arms use the exact same train/validation/test dates: the dates available to
frozen 40-day temporal Rydberg sequences. This removes the 39-row-per-split
lookback mismatch in the earlier prototype comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.targets import RV_INNOVATION_TARGET, add_rv_innovation_target
from qpitome_qrc.evaluation.transition import evaluate_transition_forecast
from qpitome_qrc.evaluation.walkforward import make_purged_walkforward_folds
from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    build_rydberg_feature_matrix,
    make_level_rate_sequence_splits,
    precompute,
)

HAR_FEATURES = ["rv_5d", "rv_10d", "rv_20d", "rv_60d"]
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]

N_ATOMS = 8
SPACING_UM = 10.0
OMEGA = 2.0 * np.pi
ENCODING_SCALE = 9.0
TOTAL_TIME = 4.0
N_PROBES = 8

_SPATIAL_CFG = RydbergQRCConfig(
    geometry="chain", chain_atoms=N_ATOMS, chain_spacing_um=SPACING_UM
)
_PRE = precompute(_SPATIAL_CFG)
_DIM = 2 ** N_ATOMS


def _sigma_x_sum() -> np.ndarray:
    out = np.zeros((_DIM, _DIM))
    idx = np.arange(_DIM)
    for q in range(N_ATOMS):
        bit = 1 << (N_ATOMS - 1 - q)
        out[idx, idx ^ bit] += 1.0
    return out


_SX_SUM = _sigma_x_sum()
_PROBE_TIMES = np.arange(1, N_PROBES + 1) * (TOTAL_TIME / N_PROBES)

TEMPORAL_CFG = RydbergQRCConfig(
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


def spatial_embeddings(x01: np.ndarray, probe_times: np.ndarray) -> np.ndarray:
    n_pair = _PRE.pair_bits.shape[1]
    out = np.empty((len(x01), len(probe_times) * (N_ATOMS + n_pair)))
    h_x = 0.5 * OMEGA * _SX_SUM
    for s in range(len(x01)):
        delta_i = ENCODING_SCALE * (x01[s] - 0.5)
        diag = _PRE.e_int - _PRE.occ_bits @ delta_i
        evals, evecs = np.linalg.eigh(h_x + np.diag(diag))
        c0 = evecs[0, :].conj()
        feats = []
        for t in probe_times:
            psi = evecs @ (np.exp(-1j * evals * t) * c0)
            probs = np.abs(psi) ** 2
            feats.extend((probs @ _PRE.occ_bits, probs @ _PRE.pair_bits))
        out[s] = np.concatenate(feats)
    return out


def fit_val_selected(f_tr, y_tr, f_va, y_va, f_te):
    scaler = StandardScaler().fit(f_tr)
    tr, va, te = (scaler.transform(x) for x in (f_tr, f_va, f_te))
    best_rmse, best_alpha = np.inf, None
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha).fit(tr, y_tr)
        rmse = float(np.sqrt(np.mean((model.predict(va) - y_va) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, alpha
    model = Ridge(alpha=best_alpha).fit(tr, y_tr)
    return model.predict(te), best_alpha, best_rmse


def reconstructed_rmse(rv20: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_future = rv20 * np.exp(y_true)
    pred_future = rv20 * np.exp(y_pred)
    return float(np.sqrt(np.mean((true_future - pred_future) ** 2)))


def align_to_dates(df: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    indexed = df.set_index("date", drop=False)
    missing = [d for d in dates if d not in indexed.index]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} sequence dates from static split")
    return indexed.loc[list(dates)].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/processed/phase2_spy_vix_volatility.csv"
    )
    parser.add_argument(
        "--out-dir", default="results/experiments/rydberg_matched_dates_v1"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    frame = add_rv_innovation_target(frame)
    folds = make_purged_walkforward_folds(
        len(frame), n_folds=5, min_train=2500, val_size=504, purge=60
    )

    rows = []
    manifest = {"target": RV_INNOVATION_TARGET, "folds": []}

    for fold in folds:
        fid = int(fold["fold"])
        raw = {
            name: frame.iloc[slice(*fold[name])].reset_index(drop=True)
            for name in ("train", "val", "test")
        }
        cleaned = {}
        required = FEATURE_COLUMNS + [RV_INNOVATION_TARGET, "rv_20d", "date"]
        for name, df in raw.items():
            sub = df.copy()
            num_cols = FEATURE_COLUMNS + [RV_INNOVATION_TARGET, "rv_20d"]
            sub[num_cols] = sub[num_cols].replace([np.inf, -np.inf], np.nan)
            cleaned[name] = sub.dropna(subset=required).reset_index(drop=True)

        seq = make_level_rate_sequence_splits(
            cleaned,
            level_col="vix_rv_spread",
            rate_col="rv_accel_log_5_20",
            target_column=RV_INNOVATION_TARGET,
            lookback_days=TEMPORAL_CFG.lookback_days,
        )

        aligned, temporal_feats = {}, {}
        for name, (xw, yy, dates) in seq.items():
            mask = np.isfinite(yy)
            dates = np.asarray(dates)[mask]
            yy = np.asarray(yy, dtype=float)[mask]
            xw = xw[mask]
            aligned[name] = align_to_dates(cleaned[name], dates)
            static_y = aligned[name][RV_INNOVATION_TARGET].to_numpy(float)
            if not np.allclose(static_y, yy, rtol=0, atol=1e-12):
                raise RuntimeError(f"Target/date alignment failed in fold {fid} {name}")
            temporal_feats[name] = build_rydberg_feature_matrix(xw, TEMPORAL_CFG)

        y = {n: aligned[n][RV_INNOVATION_TARGET].to_numpy(float) for n in aligned}
        rv20 = {n: aligned[n]["rv_20d"].to_numpy(float) for n in aligned}
        X = {n: aligned[n][FEATURE_COLUMNS].to_numpy(float) for n in aligned}

        def record(model: str, pred: np.ndarray, alpha=None, val_rmse=None, note=""):
            metrics = evaluate_transition_forecast(y["test"], pred)
            row = {
                "fold": fid,
                "model": model,
                "n_train": len(y["train"]),
                "n_val": len(y["val"]),
                "n_test": len(y["test"]),
                **metrics,
                "reconstructed_rmse": reconstructed_rmse(
                    rv20["test"], y["test"], pred
                ),
                "selected_alpha": alpha,
                "val_rmse": val_rmse,
                "note": note,
            }
            rows.append(row)
            print(
                f"fold {fid} {model:18s} n={len(pred):4d} "
                f"R2={metrics['innovation_r2']:+.4f} "
                f"RMSE={metrics['innovation_rmse']:.4f} "
                f"recon={row['reconstructed_rmse']:.4f}"
            )

        record("zero_change", np.zeros_like(y["test"]))

        har = {n: aligned[n][HAR_FEATURES].to_numpy(float) for n in aligned}
        pred, a, vr = fit_val_selected(
            har["train"], y["train"], har["val"], y["val"], har["test"]
        )
        record("har_rv_linear", pred, a, vr)

        pred, a, vr = fit_val_selected(
            X["train"], y["train"], X["val"], y["val"], X["test"]
        )
        record("full_linear", pred, a, vr)

        sc = StandardScaler().fit(X["train"])
        pca = PCA(n_components=N_ATOMS, random_state=0).fit(sc.transform(X["train"]))
        Z = {n: pca.transform(sc.transform(X[n])) for n in X}
        lo, hi = Z["train"].min(), Z["train"].max()
        Z01 = {n: np.clip((Z[n] - lo) / (hi - lo), 0.0, 1.0) for n in Z}

        pred, a, vr = fit_val_selected(
            Z["train"], y["train"], Z["val"], y["val"], Z["test"]
        )
        record("pca8_linear", pred, a, vr)

        emb = {n: spatial_embeddings(Z01[n], _PROBE_TIMES) for n in Z01}
        pred, a, vr = fit_val_selected(
            emb["train"], y["train"], emb["val"], y["val"], emb["test"]
        )
        record("quera_spatial", pred, a, vr)

        pred, a, vr = fit_val_selected(
            temporal_feats["train"], y["train"],
            temporal_feats["val"], y["val"], temporal_feats["test"]
        )
        record(
            "rydberg_temporal", pred, a, vr,
            note="frozen two-channel 40d temporal arm; exact common dates",
        )

        manifest["folds"].append(
            {
                "fold": fid,
                "train_dates": [aligned["train"]["date"].iloc[0], aligned["train"]["date"].iloc[-1]],
                "val_dates": [aligned["val"]["date"].iloc[0], aligned["val"]["date"].iloc[-1]],
                "test_dates": [aligned["test"]["date"].iloc[0], aligned["test"]["date"].iloc[-1]],
                "n_train": len(y["train"]),
                "n_val": len(y["val"]),
                "n_test": len(y["test"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    summary = (
        df.groupby("model")[["innovation_rmse", "innovation_r2", "reconstructed_rmse"]]
        .agg(["median", "mean", "std"])
    )
    summary.to_csv(out_dir / "summary.csv")
    print("\nMedian test metrics on exact common dates:")
    print(
        df.groupby("model")[["innovation_rmse", "innovation_r2", "reconstructed_rmse"]]
        .median()
        .sort_values("innovation_r2", ascending=False)
        .to_string(float_format=lambda x: f"{x:.4f}")
    )

    manifest.update(
        {
            "protocol": "canonical folds; all arms restricted to temporal 40-day sequence dates",
            "alpha_grid": ALPHA_GRID,
            "spatial": {
                "n_atoms": N_ATOMS,
                "spacing_um": SPACING_UM,
                "omega_rad_us": OMEGA,
                "encoding_scale_rad_us": ENCODING_SCALE,
                "total_time_us": TOTAL_TIME,
                "probe_times_us": _PROBE_TIMES.tolist(),
            },
            "temporal": {
                "lookback_days": TEMPORAL_CFG.lookback_days,
                "anchor_count": TEMPORAL_CFG.anchor_count,
                "total_time_us": TEMPORAL_CFG.total_time_us,
                "encoding": TEMPORAL_CFG.encoding,
            },
        }
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
