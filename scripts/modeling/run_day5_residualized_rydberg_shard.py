#!/usr/bin/env python3
"""Test whether Rydberg outputs contain signal beyond D1.

For each historical outer fold, this script:
1. generates the frozen differential-pair spatial Rydberg outputs;
2. residualizes each output block against D1 using cross-fitted Ridge models;
3. freezes the D1 logit;
4. fits only a strongly regularized logistic correction from residual outputs.

Work is sharded by unique cluster start, so rows sharing one training cutoff reuse the
same Rydberg simulation and residualizer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

REPO = Path(__file__).resolve().parents[2]
ASSAY_PATH = REPO / "scripts" / "modeling" / "run_day5_spatial_rydberg_assay_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_spatial_assay", ASSAY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load assay helpers from {ASSAY_PATH}")
assay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assay
SPEC.loader.exec_module(assay)

BLOCKS = ("occupations", "all_raw", "occ_plus_connected")
RESIDUALIZERS = ("linear", "quadratic")
RIDGE_ALPHAS = (1.0, 10.0)
OFFSET_L2 = (100.0, 1000.0)
N_SPLITS = 5


def d1_basis(X: np.ndarray, kind: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if kind == "linear":
        return X
    if kind == "quadratic":
        return PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    raise ValueError(f"Unknown residualizer kind: {kind}")


def fit_residualizer(
    X_train: np.ndarray,
    H_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted values for train and test from a standardized multioutput Ridge."""
    x_scaler = StandardScaler().fit(X_train)
    h_scaler = StandardScaler().fit(H_train)
    Xs = x_scaler.transform(X_train)
    Xts = x_scaler.transform(X_test)
    Hs = h_scaler.transform(H_train)
    model = Ridge(alpha=alpha).fit(Xs, Hs)
    fitted_train = h_scaler.inverse_transform(model.predict(Xs))
    fitted_test = h_scaler.inverse_transform(model.predict(Xts))
    return fitted_train, fitted_test


def cross_fitted_residuals(
    X_train: np.ndarray,
    H_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    n_splits: int = N_SPLITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fit train residuals and fit the held-out residual with the full training set."""
    n = len(X_train)
    splits = min(n_splits, n)
    if splits < 2:
        raise ValueError("At least two training rows are required")
    predicted_train = np.empty_like(H_train, dtype=float)
    kfold = KFold(n_splits=splits, shuffle=False)
    for fit_idx, valid_idx in kfold.split(X_train):
        _, predicted_valid = fit_residualizer(
            X_train[fit_idx], H_train[fit_idx], X_train[valid_idx], alpha
        )
        predicted_train[valid_idx] = predicted_valid
    _, predicted_test = fit_residualizer(X_train, H_train, X_test, alpha)
    return H_train - predicted_train, H_train[:0] if len(X_test) == 0 else None


def residualize_train_test(
    X_train: np.ndarray,
    H_train: np.ndarray,
    X_test: np.ndarray,
    H_test: np.ndarray,
    alpha: float,
    n_splits: int = N_SPLITS,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X_train)
    splits = min(n_splits, n)
    predicted_train = np.empty_like(H_train, dtype=float)
    kfold = KFold(n_splits=splits, shuffle=False)
    for fit_idx, valid_idx in kfold.split(X_train):
        _, predicted_valid = fit_residualizer(
            X_train[fit_idx], H_train[fit_idx], X_train[valid_idx], alpha
        )
        predicted_train[valid_idx] = predicted_valid
    _, predicted_test = fit_residualizer(X_train, H_train, X_test, alpha)
    return H_train - predicted_train, H_test - predicted_test


def residual_diagnostics(H: np.ndarray, R: np.ndarray) -> dict[str, float]:
    total = float(np.sum((H - H.mean(axis=0, keepdims=True)) ** 2))
    residual = float(np.sum(R**2))
    explained = 1.0 - residual / total if total > 1e-15 else float("nan")
    return {
        "fraction_output_variance_removed": float(explained),
        "residual_rms": float(np.sqrt(np.mean(R**2))),
    }


def run_shard(
    frame: pd.DataFrame,
    shard_index: int,
    num_shards: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = assay.rydberg_config()
    eligible = assay.eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))
    selected = [group for position, group in enumerate(fold_groups) if position % num_shards == shard_index]

    predictions: list[dict] = []
    diagnostics: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(selected, start=1):
        train = frame[frame["landmark_date"] < cluster_start]
        test_indices = test_group.index.to_numpy(int)
        combined = pd.concat([train, frame.loc[test_indices]], axis=0)

        patterns = assay.differential_patterns(
            train[assay.STATIC].to_numpy(float),
            combined[assay.STATIC].to_numpy(float),
        )
        features = assay.build_local_detuning_feature_matrix(patterns, config)
        blocks = assay.split_blocks(features)

        d1_train = train[assay.D1].to_numpy(float)
        d1_test = frame.loc[test_indices, assay.D1].to_numpy(float)
        y_train = train["y_recovery"].to_numpy(int)
        d1_model = assay.logistic_pipeline(1.0)
        d1_model.fit(d1_train, y_train)
        p_train = d1_model.predict_proba(d1_train)[:, 1]
        p_test = d1_model.predict_proba(d1_test)[:, 1]
        offset_train = assay.logit(p_train)
        offset_test = assay.logit(p_test)

        fold_predictions: dict[str, np.ndarray] = {"D1": p_test}

        for block_name in BLOCKS:
            block = blocks[block_name]
            H_train = block[: len(train)]
            H_test = block[len(train) :]
            for residualizer in RESIDUALIZERS:
                X_train = d1_basis(d1_train, residualizer)
                X_test = d1_basis(d1_test, residualizer)
                for ridge_alpha in RIDGE_ALPHAS:
                    R_train, R_test = residualize_train_test(
                        X_train, H_train, X_test, H_test, ridge_alpha
                    )
                    diag = residual_diagnostics(H_train, R_train)
                    diagnostics.append({
                        "cluster_start": cluster_start,
                        "block": block_name,
                        "residualizer": residualizer,
                        "ridge_alpha": ridge_alpha,
                        "n_train": int(len(train)),
                        "n_test": int(len(test_indices)),
                        **diag,
                    })
                    for l2 in OFFSET_L2:
                        name = f"offset_resid_{block_name}_{residualizer}_ridge{ridge_alpha:g}_l2{int(l2)}"
                        values = []
                        for row_number in range(len(test_indices)):
                            values.append(
                                assay.fit_offset_predict(
                                    R_train,
                                    y_train,
                                    R_test[[row_number]],
                                    offset_train,
                                    float(offset_test[row_number]),
                                    l2,
                                )
                            )
                        fold_predictions[name] = np.asarray(values, dtype=float)

        for local_row, row_index in enumerate(test_indices):
            row = frame.loc[row_index]
            record = {
                "row_id": int(row_index),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "cluster_start": cluster_start,
                "y": int(row["y_recovery"]),
                "shard_index": shard_index,
            }
            for name, values in fold_predictions.items():
                record[name] = float(values[local_row])
            predictions.append(record)

        print(
            f"shard {shard_index}/{num_shards}: fold {fold_number}/{len(selected)} "
            f"cluster_start={pd.Timestamp(cluster_start).date()} train={len(train)} test={len(test_indices)}",
            flush=True,
        )

    return pd.DataFrame(predictions), pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")

    frame = assay.load_frame(args.path_panel, args.clusters)
    predictions, diagnostics = run_shard(frame, args.shard_index, args.num_shards)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / f"predictions_shard_{args.shard_index}.csv", index=False)
    diagnostics.to_csv(args.outdir / f"residual_diagnostics_shard_{args.shard_index}.csv", index=False)
    manifest = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "blocks": BLOCKS,
        "residualizers": RESIDUALIZERS,
        "ridge_alphas": RIDGE_ALPHAS,
        "offset_l2": OFFSET_L2,
        "n_predictions": int(len(predictions)),
    }
    (args.outdir / f"manifest_shard_{args.shard_index}.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved shard {args.shard_index} to {args.outdir}")


if __name__ == "__main__":
    main()
