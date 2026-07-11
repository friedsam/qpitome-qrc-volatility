#!/usr/bin/env python3
"""Reusable day-5 feature assay driven by a JSON feature specification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

REPO = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


input_audit = load_module("day5_input_audit", REPO / "scripts/modeling/run_day5_input_audit.py")
protected = load_module("day5_protected_input", REPO / "scripts/modeling/run_day5_protected_input_residual.py")
base = protected.base

RIDGE_ALPHA = 10.0
OFFSET_L2 = 100.0
TANH_WIDTH = 32
RNG_SEED = 20260711
INNER_MIN_TRAIN = 10
MIN_CORRECTION_TRAIN = 20


def prepare_features(frame: pd.DataFrame, preparation: str | None) -> pd.DataFrame:
    if preparation in (None, "none"):
        return frame.copy()
    if preparation == "path_shape":
        return input_audit.add_path_shape_features(frame)
    raise ValueError(f"Unknown preparation: {preparation}")


def historical_crossfit_logits(train: pd.DataFrame, baseline: list[str]) -> tuple[np.ndarray, np.ndarray]:
    logits = np.full(len(train), np.nan, dtype=float)
    for cluster_start, group in train.groupby("cluster_start", sort=True):
        positions = train.index.get_indexer(group.index)
        prior = train[train["landmark_date"] < cluster_start]
        if len(prior) < INNER_MIN_TRAIN or prior["y_recovery"].nunique() < 2:
            continue
        model = base.assay.logistic_pipeline(1.0)
        model.fit(prior[baseline].to_numpy(float), prior["y_recovery"].to_numpy(int))
        p = model.predict_proba(group[baseline].to_numpy(float))[:, 1]
        logits[positions] = base.assay.logit(p)
    valid = np.isfinite(logits)
    return np.flatnonzero(valid), logits[valid]


def nonlinear_maps(R_train: np.ndarray, R_test: np.ndarray, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scaler = StandardScaler().fit(R_train)
    Z_train = scaler.transform(R_train)
    Z_test = scaler.transform(R_test)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    P_train = poly.fit_transform(Z_train)
    P_test = poly.transform(Z_test)
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=1.0 / np.sqrt(max(1, Z_train.shape[1])), size=(Z_train.shape[1], TANH_WIDTH))
    b = rng.uniform(-1.0, 1.0, size=TANH_WIDTH)
    return {
        "poly2": (P_train, P_test),
        "tanh32": (np.tanh(Z_train @ W + b), np.tanh(Z_test @ W + b)),
    }


def score_deltas(frame: pd.DataFrame, baseline_col: str, model_col: str) -> dict[str, float | int | str]:
    use = frame[["y", baseline_col, model_col, "cluster_id"]].dropna().copy()
    y = use["y"].to_numpy(int)
    p0 = np.clip(use[baseline_col].to_numpy(float), 1e-8, 1 - 1e-8)
    p1 = np.clip(use[model_col].to_numpy(float), 1e-8, 1 - 1e-8)
    ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    br0 = (p0 - y) ** 2
    br1 = (p1 - y) ** 2
    use["dll"] = ll1 - ll0
    use["dbr"] = br1 - br0
    cluster = use.groupby("cluster_id")[["dll", "dbr"]].mean()
    return {
        "model": model_col,
        "baseline": baseline_col,
        "n": int(len(use)),
        "n_clusters": int(use["cluster_id"].nunique()),
        "delta_logloss": float(np.mean(ll1 - ll0)),
        "delta_brier": float(np.mean(br1 - br0)),
        "cluster_mean_delta_logloss": float(cluster["dll"].mean()),
        "cluster_mean_delta_brier": float(cluster["dbr"].mean()),
    }


def run(frame: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = prepare_features(frame, spec.get("preparation"))
    blocks = spec["blocks"]
    eligible = base.assay.eligible_rows(frame)
    fold_groups = list(frame.loc[eligible].groupby("cluster_start", sort=True))
    predictions: list[dict] = []
    reconstruction: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(fold_groups, start=1):
        train = frame[frame["landmark_date"] < cluster_start].copy()
        test = frame.loc[test_group.index].copy()
        y_train = train["y_recovery"].to_numpy(int)
        fold_outputs: dict[str, np.ndarray] = {}

        for block_pos, (block_name, block_spec) in enumerate(blocks.items()):
            features = block_spec["features"]
            baseline = block_spec["baseline_features"]
            B_train = train[baseline].to_numpy(float)
            B_test = test[baseline].to_numpy(float)
            H_train = train[features].to_numpy(float)
            H_test = test[features].to_numpy(float)

            baseline_model = base.assay.logistic_pipeline(1.0)
            baseline_model.fit(B_train, y_train)
            p_base = baseline_model.predict_proba(B_test)[:, 1]
            baseline_name = f"baseline__{block_name}"
            fold_outputs[baseline_name] = p_base

            raw_model = base.assay.logistic_pipeline(1.0)
            raw_model.fit(np.column_stack([B_train, H_train]), y_train)
            fold_outputs[f"raw__{block_name}"] = raw_model.predict_proba(np.column_stack([B_test, H_test]))[:, 1]

            recon = Ridge(alpha=RIDGE_ALPHA)
            Bs = StandardScaler().fit(B_train)
            Hs = StandardScaler().fit(H_train)
            recon.fit(Bs.transform(B_train), Hs.transform(H_train))
            pred_h = np.asarray(recon.predict(Bs.transform(B_test)), dtype=float)
            if pred_h.ndim == 1:
                pred_h = pred_h.reshape(-1, 1)
            pred_h = Hs.inverse_transform(pred_h)
            for local_row, row_index in enumerate(test.index):
                for feature_pos, feature in enumerate(features):
                    reconstruction.append({
                        "row_id": int(row_index), "cluster_id": test.loc[row_index, "cluster_id"],
                        "block": block_name, "feature": feature,
                        "actual": float(H_test[local_row, feature_pos]),
                        "predicted_from_baseline": float(pred_h[local_row, feature_pos]),
                    })

            X_train = base.d1_basis(B_train, "quadratic")
            X_test = base.d1_basis(B_test, "quadratic")
            R_train, R_test = protected.residualize_train_test_safe(X_train, H_train, X_test, H_test, RIDGE_ALPHA)
            cf_positions, cf_logits = historical_crossfit_logits(train, baseline)
            if len(cf_positions) < MIN_CORRECTION_TRAIN or np.unique(y_train[cf_positions]).size < 2:
                raise RuntimeError(f"Insufficient cross-fitted rows for {cluster_start} / {block_name}")
            offset_test = base.assay.logit(p_base)

            maps = {"linear": (R_train, R_test), **nonlinear_maps(R_train, R_test, RNG_SEED + 1000 * block_pos + fold_number)}
            for map_pos, (map_name, (M_train, M_test)) in enumerate(maps.items()):
                corrected = []
                for row_number in range(len(test)):
                    corrected.append(base.assay.fit_offset_predict(
                        M_train[cf_positions], y_train[cf_positions], M_test[[row_number]],
                        cf_logits, float(offset_test[row_number]), OFFSET_L2,
                    ))
                fold_outputs[f"protected_{map_name}__{block_name}"] = np.asarray(corrected)

                rng = np.random.default_rng(RNG_SEED + 100000 + 1000 * block_pos + 10 * map_pos + fold_number)
                G_train = rng.normal(size=M_train.shape)
                G_test = rng.normal(size=M_test.shape)
                gaussian = []
                for row_number in range(len(test)):
                    gaussian.append(base.assay.fit_offset_predict(
                        G_train[cf_positions], y_train[cf_positions], G_test[[row_number]],
                        cf_logits, float(offset_test[row_number]), OFFSET_L2,
                    ))
                fold_outputs[f"gaussian_{map_name}__{block_name}"] = np.asarray(gaussian)

        for local_row, row_index in enumerate(test.index):
            row = test.loc[row_index]
            record = {
                "row_id": int(row_index), "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]), "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"], "cluster_start": cluster_start,
                "y": int(row["y_recovery"]),
            }
            for name, values in fold_outputs.items():
                record[name] = float(values[local_row])
            predictions.append(record)

        print(f"fold {fold_number}/{len(fold_groups)} cluster_start={pd.Timestamp(cluster_start).date()} train={len(train)} test={len(test)}", flush=True)

    return pd.DataFrame(predictions).sort_values("row_id").reset_index(drop=True), pd.DataFrame(reconstruction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-spec", type=Path, required=True)
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.feature_spec.read_text())
    frame = base.assay.load_frame(args.path_panel, args.clusters)
    predictions, reconstruction = run(frame, spec)

    summaries = []
    for block_name in spec["blocks"]:
        baseline_col = f"baseline__{block_name}"
        for model_col in [c for c in predictions.columns if c.endswith(f"__{block_name}") and c != baseline_col]:
            summaries.append(score_deltas(predictions, baseline_col, model_col))
    summary = pd.DataFrame(summaries).sort_values(["delta_logloss", "delta_brier"])

    recon_rows = []
    for (block, feature), group in reconstruction.groupby(["block", "feature"], sort=True):
        recon_rows.append({
            "block": block, "feature": feature, "n": int(len(group)),
            "r2_from_baseline": float(r2_score(group["actual"], group["predicted_from_baseline"])),
            "mae_from_baseline": float(np.mean(np.abs(group["actual"] - group["predicted_from_baseline"]))),
        })
    recon_summary = pd.DataFrame(recon_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    summary.to_csv(args.outdir / "score_summary.csv", index=False)
    reconstruction.to_csv(args.outdir / "reconstruction_predictions.csv", index=False)
    recon_summary.to_csv(args.outdir / "reconstruction_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps({
        "feature_spec": spec, "ridge_alpha": RIDGE_ALPHA, "offset_l2": OFFSET_L2,
        "tanh_width": TANH_WIDTH, "rng_seed": RNG_SEED,
        "n_predictions": int(len(predictions)),
    }, indent=2))

    print("\nStandard feature assay ranked by log-loss delta (negative = better)")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nHeld-out feature reconstruction from each declared baseline")
    print(recon_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
