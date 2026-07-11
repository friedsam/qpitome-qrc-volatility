#!/usr/bin/env python3
"""Fixed nonlinear classical residual assay for selected path features."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


protected = load_module(
    "day5_protected_input",
    REPO / "scripts" / "modeling" / "run_day5_protected_input_residual.py",
)
base = protected.base
crossfit = protected.crossfit

FEATURE_BLOCKS = protected.FEATURE_BLOCKS
OFFSET_L2 = 100.0
RIDGE_ALPHA = 10.0
TANH_WIDTH = 32
RNG_SEED = 20260711


def nonlinear_maps(
    R_train: np.ndarray,
    R_test: np.ndarray,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scaler = StandardScaler().fit(R_train)
    Z_train = scaler.transform(R_train)
    Z_test = scaler.transform(R_test)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    P_train = poly.fit_transform(Z_train)
    P_test = poly.transform(Z_test)

    rng = np.random.default_rng(seed)
    W = rng.normal(scale=1.0 / np.sqrt(max(1, Z_train.shape[1])), size=(Z_train.shape[1], TANH_WIDTH))
    b = rng.uniform(-1.0, 1.0, size=TANH_WIDTH)
    T_train = np.tanh(Z_train @ W + b)
    T_test = np.tanh(Z_test @ W + b)

    return {
        "poly2": (P_train, P_test),
        "tanh32": (T_train, T_test),
    }


def gaussian_control(
    train_shape: tuple[int, int],
    test_shape: tuple[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=train_shape), rng.normal(size=test_shape)


def run(frame: pd.DataFrame) -> pd.DataFrame:
    frame = protected.input_audit.add_path_shape_features(frame)
    eligible = base.assay.eligible_rows(frame)
    fold_groups = list(frame.loc[eligible].groupby("cluster_start", sort=True))
    records: list[dict] = []

    for fold_number, (cluster_start, test_group) in enumerate(fold_groups, start=1):
        train = frame[frame["landmark_date"] < cluster_start].copy()
        test = frame.loc[test_group.index].copy()
        y_train = train["y_recovery"].to_numpy(int)
        d1_train = train[base.assay.D1].to_numpy(float)
        d1_test = test[base.assay.D1].to_numpy(float)
        X_train = base.d1_basis(d1_train, "quadratic")
        X_test = base.d1_basis(d1_test, "quadratic")

        cf_positions, cf_logits = crossfit.historical_crossfit_d1_logits(train)
        if len(cf_positions) < crossfit.MIN_CORRECTION_TRAIN or np.unique(y_train[cf_positions]).size < 2:
            raise RuntimeError(f"Insufficient cross-fitted rows for {cluster_start}")

        d1_model = base.assay.logistic_pipeline(1.0)
        d1_model.fit(d1_train, y_train)
        p_test_d1 = d1_model.predict_proba(d1_test)[:, 1]
        offset_test = base.assay.logit(p_test_d1)
        fold_predictions: dict[str, np.ndarray] = {"D1": p_test_d1}

        fold_seed = RNG_SEED + int(pd.Timestamp(cluster_start).value % 2**31)
        for block_pos, (block_name, columns) in enumerate(FEATURE_BLOCKS.items()):
            H_train = train[columns].to_numpy(float)
            H_test = test[columns].to_numpy(float)
            R_train, R_test = protected.residualize_train_test_safe(
                X_train, H_train, X_test, H_test, RIDGE_ALPHA
            )
            maps = nonlinear_maps(R_train, R_test, fold_seed + 1000 * block_pos)

            for map_pos, (map_name, (M_train, M_test)) in enumerate(maps.items()):
                corrected = []
                for row_number in range(len(test)):
                    corrected.append(base.assay.fit_offset_predict(
                        M_train[cf_positions], y_train[cf_positions], M_test[[row_number]],
                        cf_logits, float(offset_test[row_number]), OFFSET_L2,
                    ))
                fold_predictions[f"nonlinear_{map_name}_{block_name}"] = np.asarray(corrected)

                G_train, G_test = gaussian_control(
                    M_train.shape, M_test.shape,
                    fold_seed + 100000 + 1000 * block_pos + map_pos,
                )
                gaussian = []
                for row_number in range(len(test)):
                    gaussian.append(base.assay.fit_offset_predict(
                        G_train[cf_positions], y_train[cf_positions], G_test[[row_number]],
                        cf_logits, float(offset_test[row_number]), OFFSET_L2,
                    ))
                fold_predictions[f"gaussian_{map_name}_{block_name}"] = np.asarray(gaussian)

        for local_row, row_index in enumerate(test.index):
            row = test.loc[row_index]
            record = {
                "row_id": int(row_index),
                "market_key": row["market_key"],
                "episode_id": int(row["episode_id"]),
                "cluster_id": row["cluster_id"],
                "landmark_date": row["landmark_date"],
                "cluster_start": cluster_start,
                "y": int(row["y_recovery"]),
                "n_correction_train": int(len(cf_positions)),
            }
            for name, values in fold_predictions.items():
                record[name] = float(values[local_row])
            records.append(record)

        print(
            f"fold {fold_number}/{len(fold_groups)} cluster_start={pd.Timestamp(cluster_start).date()} "
            f"train={len(train)} correction_train={len(cf_positions)} test={len(test)}",
            flush=True,
        )

    return pd.DataFrame(records).sort_values("row_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    frame = base.assay.load_frame(args.path_panel, args.clusters)
    predictions = run(frame)
    models = [column for column in predictions.columns if column.startswith("nonlinear_") or column.startswith("gaussian_")]
    summary = pd.DataFrame([protected.score_deltas(predictions, model) for model in models]).sort_values(
        ["delta_logloss", "delta_brier"]
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "nonlinear_input_predictions.csv", index=False)
    summary.to_csv(args.outdir / "nonlinear_input_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps({
        "feature_blocks": FEATURE_BLOCKS,
        "maps": ["poly2", "tanh32"],
        "tanh_width": TANH_WIDTH,
        "ridge_alpha": RIDGE_ALPHA,
        "offset_l2": OFFSET_L2,
        "rng_seed": RNG_SEED,
        "n_predictions": int(len(predictions)),
    }, indent=2))

    print("\nNonlinear protected path features ranked by log-loss delta (negative = better)")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
