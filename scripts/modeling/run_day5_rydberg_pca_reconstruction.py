#!/usr/bin/env python3
"""Analyze which Rydberg-output PCA components reconstruct D1 state and logit.

The diagnostic is leakage-safe at the outer historical fold level. For each unique
cluster start, it fits the input scaler, Rydberg-output scaler, PCA, D1 model, and
Ridge reconstruction models on earlier episodes only, then predicts every held-out
row in that cluster.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
ASSAY_PATH = REPO / "scripts" / "modeling" / "run_day5_spatial_rydberg_assay_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_spatial_assay", ASSAY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load assay helpers from {ASSAY_PATH}")
assay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assay
SPEC.loader.exec_module(assay)

BLOCKS = ("occupations", "all_raw", "occ_plus_connected")
RIDGE_ALPHA = 10.0
MAX_COMPONENTS = 15


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def fit_predict_ridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit a standardized multi-output Ridge model and return test predictions."""
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)
    y_scaler = StandardScaler().fit(Y_train)
    Y_train_s = y_scaler.transform(Y_train)
    model = Ridge(alpha=alpha).fit(X_train_s, Y_train_s)
    return y_scaler.inverse_transform(model.predict(X_test_s))


def run_diagnostic(
    frame: pd.DataFrame,
    max_components: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = assay.rydberg_config()
    eligible = assay.eligible_rows(frame)
    eligible_frame = frame.loc[eligible].copy()
    fold_groups = list(eligible_frame.groupby("cluster_start", sort=True))

    prediction_rows: list[dict] = []
    variance_rows: list[dict] = []
    loading_rows: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(fold_groups, start=1):
        train = frame[frame["landmark_date"] < cluster_start]
        if len(train) < assay.MIN_TRAIN or train["y_recovery"].nunique() < 2:
            continue

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
        d1_model = assay.logistic_pipeline(1.0)
        d1_model.fit(d1_train, train["y_recovery"].to_numpy(int))
        d1_logit_train = d1_model.decision_function(d1_train)
        d1_logit_test = d1_model.decision_function(d1_test)
        targets_train = np.column_stack([d1_train, d1_logit_train])
        targets_test = np.column_stack([d1_test, d1_logit_test])
        target_names = [*assay.D1, "D1_logit"]

        for block_name in BLOCKS:
            block = blocks[block_name]
            H_train = block[: len(train)]
            H_test = block[len(train) :]
            scaler = StandardScaler().fit(H_train)
            H_train_s = scaler.transform(H_train)
            H_test_s = scaler.transform(H_test)
            rank = min(max_components, H_train_s.shape[0] - 1, H_train_s.shape[1])
            pca = PCA(n_components=rank, svd_solver="full").fit(H_train_s)
            Z_train = pca.transform(H_train_s)
            Z_test = pca.transform(H_test_s)

            cumulative = np.cumsum(pca.explained_variance_ratio_)
            for component in range(rank):
                variance_rows.append({
                    "cluster_start": cluster_start,
                    "block": block_name,
                    "component": component + 1,
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[component]),
                    "cumulative_explained_variance": float(cumulative[component]),
                    "n_train": int(len(train)),
                    "n_test": int(len(test_indices)),
                })
                for feature_index, loading in enumerate(pca.components_[component]):
                    loading_rows.append({
                        "cluster_start": cluster_start,
                        "block": block_name,
                        "component": component + 1,
                        "feature_index": feature_index,
                        "loading": float(loading),
                    })

            model_specs: list[tuple[str, int, np.ndarray, np.ndarray]] = []
            for component in range(rank):
                model_specs.append(("individual", component + 1, Z_train[:, [component]], Z_test[:, [component]]))
                model_specs.append(("cumulative", component + 1, Z_train[:, : component + 1], Z_test[:, : component + 1]))

            for mode, component_count, X_train, X_test in model_specs:
                prediction = fit_predict_ridge(X_train, targets_train, X_test, ridge_alpha)
                for local_row, row_index in enumerate(test_indices):
                    base = {
                        "row_id": int(row_index),
                        "cluster_start": cluster_start,
                        "market_key": frame.loc[row_index, "market_key"],
                        "episode_id": int(frame.loc[row_index, "episode_id"]),
                        "block": block_name,
                        "mode": mode,
                        "component_count": component_count,
                    }
                    for target_index, target_name in enumerate(target_names):
                        prediction_rows.append({
                            **base,
                            "target": target_name,
                            "y_true": float(targets_test[local_row, target_index]),
                            "y_pred": float(prediction[local_row, target_index]),
                        })

        print(
            f"fold {fold_number}/{len(fold_groups)} cluster_start={pd.Timestamp(cluster_start).date()} "
            f"train={len(train)} test={len(test_indices)}",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    variance = pd.DataFrame(variance_rows)
    loadings = pd.DataFrame(loading_rows)
    return predictions, variance, loadings


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["block", "mode", "component_count", "target"]
    for key, group in predictions.groupby(group_cols, sort=True):
        block, mode, component_count, target = key
        rows.append({
            "block": block,
            "mode": mode,
            "component_count": int(component_count),
            "target": target,
            "n": int(len(group)),
            "r2": safe_r2(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float)),
            "mae": float(np.mean(np.abs(group["y_true"] - group["y_pred"]))),
            "rmse": float(np.sqrt(np.mean((group["y_true"] - group["y_pred"]) ** 2))),
        })
    return pd.DataFrame(rows)


def summarize_loadings(loadings: pd.DataFrame) -> pd.DataFrame:
    if loadings.empty:
        return loadings
    aggregate = (
        loadings.assign(abs_loading=lambda x: np.abs(x["loading"]))
        .groupby(["block", "component", "feature_index"], as_index=False)["abs_loading"]
        .median()
    )
    return aggregate.sort_values(["block", "component", "abs_loading"], ascending=[True, True, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--max-components", type=int, default=MAX_COMPONENTS)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    frame = assay.load_frame(args.path_panel, args.clusters)
    predictions, variance, loadings = run_diagnostic(frame, args.max_components, args.ridge_alpha)
    summary = summarize(predictions)
    loading_summary = summarize_loadings(loadings)

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "pca_reconstruction_predictions.csv", index=False)
    summary.to_csv(args.outdir / "pca_reconstruction_summary.csv", index=False)
    variance.to_csv(args.outdir / "pca_explained_variance_by_fold.csv", index=False)
    loadings.to_csv(args.outdir / "pca_loadings_by_fold.csv", index=False)
    loading_summary.to_csv(args.outdir / "pca_loading_summary.csv", index=False)
    manifest = {
        "max_components": args.max_components,
        "ridge_alpha": args.ridge_alpha,
        "blocks": BLOCKS,
        "n_prediction_rows": int(len(predictions)),
        "n_outer_rows": int(predictions["row_id"].nunique()),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nCumulative D1-logit reconstruction")
    display = summary[(summary["target"] == "D1_logit") & (summary["mode"] == "cumulative")]
    print(display.sort_values(["block", "component_count"]).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved outputs to {args.outdir}")


if __name__ == "__main__":
    main()
