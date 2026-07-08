#!/usr/bin/env python3
"""Targeted ESN follow-up for rv_innovation_20d.

Freeze each fold's reservoir configuration from the prior PCA6 validation search.
Select only representation dimension and ridge alpha on validation RMSE.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.numpy_esn import make_esn_weights
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
from qpitome_qrc.evaluation.walkforward import (
    make_purged_walkforward_folds,
    slice_fold_frames,
)

SPLITS = ("train", "val", "test")
HAR = ("rv_5d", "rv_10d", "rv_20d", "rv_60d")


def numbers(text, cast):
    return [cast(x) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/phase2_spy_vix_volatility.csv"),
    )
    p.add_argument(
        "--reservoir-config",
        type=Path,
        default=Path("configs/esn_transition_fixed_reservoirs.json"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/canonical/transition_v1/esn"),
    )
    p.add_argument("--tag", default="esn_transition_v1")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--representation-grid", default="6,10,16,26")
    p.add_argument("--alphas", default="10,30,100,300,1000")
    p.add_argument("--robust-seeds", default="7,42,123")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--only-folds", nargs="*", type=int)
    return p.parse_args()


def windows(values, lookback):
    raw = np.lib.stride_tricks.sliding_window_view(
        values, lookback, axis=0
    )
    return np.transpose(raw, (0, 2, 1)).copy()


def states(X, W_in, W, leak):
    h = np.zeros((len(X), W.shape[0]))
    for t in range(X.shape[1]):
        candidate = np.tanh(X[:, t, :] @ W_in.T + h @ W.T)
        h = (1.0 - leak) * h + leak * candidate
    return np.concatenate([h, X[:, -1, :]], axis=1)


def fit_ridge(H, y, alpha):
    scaler = StandardScaler().fit(H["train"])
    model = Ridge(alpha=alpha).fit(
        scaler.transform(H["train"]),
        y["train"],
    )
    return {
        split: model.predict(scaler.transform(H[split]))
        for split in SPLITS
    }


def select_alpha(H, y, alphas):
    best = None
    for alpha in alphas:
        scores = fit_ridge(H, y, alpha)
        rmse = evaluate_transition_forecast(
            y["val"], scores["val"]
        )["innovation_rmse"]
        candidate = (rmse, alpha, scores)
        if best is None or rmse < best[0]:
            best = candidate
    return best


def representation(frames, dim):
    scaler = StandardScaler().fit(
        frames["train"][FEATURE_COLUMNS]
    )
    scaled = {
        split: scaler.transform(frames[split][FEATURE_COLUMNS])
        for split in SPLITS
    }

    if dim == len(FEATURE_COLUMNS):
        return scaled

    pca = PCA(n_components=dim, random_state=42).fit(
        scaled["train"]
    )
    return {
        split: pca.transform(scaled[split])
        for split in SPLITS
    }


def metric_row(
    model,
    fold,
    y,
    scores,
    refs,
    levels,
    **extra,
):
    row = {"fold": fold, "model": model, **extra}

    for split in ("val", "test"):
        row.update({
            f"{split}_{key}": value
            for key, value in evaluate_transition_forecast(
                y[split], scores[split]
            ).items()
        })

        reconstructed = reconstruct_future_rv(
            refs[split], scores[split]
        )
        level_metrics = asdict(
            evaluate_volatility_forecast(
                levels[split], reconstructed
            )
        )
        row.update({
            f"{split}_reconstructed_{key}": value
            for key, value in level_metrics.items()
        })

    return row


def main():
    args = parse_args()

    dims = numbers(args.representation_grid, int)
    alphas = numbers(args.alphas, float)
    seeds = numbers(args.robust_seeds, int)

    if any(
        dim < 1 or dim > len(FEATURE_COLUMNS)
        for dim in dims
    ):
        raise ValueError(
            f"representation dimensions must be 1..{len(FEATURE_COLUMNS)}"
        )

    frozen = json.loads(
        args.reservoir_config.read_text(encoding="utf-8")
    )
    search_seed = int(frozen["search_seed"])

    df = add_rv_innovation_target(
        pd.read_csv(args.data)
        .sort_values("date")
        .reset_index(drop=True)
    )

    folds = make_purged_walkforward_folds(
        len(df),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )

    if args.only_folds:
        wanted = set(args.only_folds)
        folds = [
            fold
            for fold in folds
            if int(fold["fold"]) in wanted
        ]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    predictions = []

    for fold in folds:
        fold_id = int(fold["fold"])
        cfg = frozen["folds"][str(fold_id)]

        print(
            f"\n=== fold {fold_id}; "
            f"frozen reservoir={cfg} ==="
        )

        frames = slice_fold_frames(df, fold)
        aligned = {
            split: frames[split]
            .iloc[args.lookback - 1 :]
            .reset_index(drop=True)
            for split in SPLITS
        }

        y = {
            split: aligned[split][RV_INNOVATION_TARGET]
            .to_numpy(float)
            for split in SPLITS
        }
        refs = {
            split: aligned[split][RV_REFERENCE_COLUMN]
            .to_numpy(float)
            for split in SPLITS
        }
        levels = {
            split: aligned[split][RV_LEVEL_TARGET]
            .to_numpy(float)
            for split in SPLITS
        }
        dates = {
            split: aligned[split]["date"].to_numpy()
            for split in SPLITS
        }

        zero = {
            split: np.zeros_like(y[split])
            for split in SPLITS
        }
        rows.append(
            metric_row(
                "zero_change",
                fold_id,
                y,
                zero,
                refs,
                levels,
            )
        )

        for name, columns in (
            ("har_rv_linear", HAR),
            ("full_linear", FEATURE_COLUMNS),
        ):
            H = {
                split: aligned[split][list(columns)]
                .to_numpy(float)
                for split in SPLITS
            }
            _, alpha, scores = select_alpha(
                H, y, alphas
            )
            rows.append(
                metric_row(
                    name,
                    fold_id,
                    y,
                    scores,
                    refs,
                    levels,
                    selected_alpha=alpha,
                )
            )

        best_linear = None
        best_esn = None

        for dim in dims:
            z = representation(frames, dim)

            last = {
                split: z[split][args.lookback - 1 :]
                for split in SPLITS
            }
            linear_candidate = select_alpha(
                last, y, alphas
            )
            if (
                best_linear is None
                or linear_candidate[0] < best_linear[0]
            ):
                best_linear = (
                    *linear_candidate,
                    dim,
                )

            X = {
                split: windows(
                    z[split], args.lookback
                )
                for split in SPLITS
            }

            W_in, W = make_esn_weights(
                z["train"].shape[1],
                cfg["n"],
                cfg["sr"],
                cfg["inp"],
                search_seed,
            )

            H = {
                split: states(
                    X[split],
                    W_in,
                    W,
                    cfg["leak"],
                )
                for split in SPLITS
            }

            esn_candidate = select_alpha(
                H, y, alphas
            )
            if (
                best_esn is None
                or esn_candidate[0] < best_esn[0]
            ):
                best_esn = (
                    *esn_candidate,
                    dim,
                    z,
                    X,
                )

        _, alpha, scores, dim = best_linear
        rows.append(
            metric_row(
                "representation_linear_selected",
                fold_id,
                y,
                scores,
                refs,
                levels,
                selected_dim=dim,
                selected_alpha=alpha,
            )
        )

        _, alpha, scores, dim, z, X = best_esn

        seed_r2 = {}
        for seed in seeds:
            W_in, W = make_esn_weights(
                z["train"].shape[1],
                cfg["n"],
                cfg["sr"],
                cfg["inp"],
                seed,
            )

            H = {
                split: states(
                    X[split],
                    W_in,
                    W,
                    cfg["leak"],
                )
                for split in SPLITS
            }

            seed_scores = fit_ridge(
                H, y, alpha
            )

            seed_r2[str(seed)] = (
                evaluate_transition_forecast(
                    y["test"],
                    seed_scores["test"],
                )["innovation_r2"]
            )

            if seed == search_seed:
                scores = seed_scores

        rows.append(
            metric_row(
                "esn_selected",
                fold_id,
                y,
                scores,
                refs,
                levels,
                selected_dim=dim,
                selected_alpha=alpha,
                selected_seed=search_seed,
                seed_test_r2=json.dumps(
                    seed_r2,
                    sort_keys=True,
                ),
                **cfg,
            )
        )

        for split in ("val", "test"):
            level_pred = reconstruct_future_rv(
                refs[split],
                scores[split],
            )
            for values in zip(
                dates[split],
                y[split],
                scores[split],
                levels[split],
                level_pred,
                strict=True,
            ):
                (
                    date,
                    truth,
                    pred,
                    level_true,
                    level_hat,
                ) = values

                predictions.append({
                    "fold": fold_id,
                    "split": split,
                    "date": date,
                    "model": "esn_selected",
                    "innovation_true": truth,
                    "innovation_pred": pred,
                    "future_rv_true": level_true,
                    "predicted_future_rv": level_hat,
                })

        print(f"Completed fold {fold_id}")

    metrics = pd.DataFrame(rows)

    metrics_path = (
        args.out_dir
        / f"esn_transition_metrics_{args.tag}.csv"
    )
    predictions_path = (
        args.out_dir
        / f"esn_transition_predictions_{args.tag}.csv"
    )
    summary_path = (
        args.out_dir
        / f"esn_transition_summary_{args.tag}.json"
    )

    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(predictions).to_csv(
        predictions_path,
        index=False,
    )

    summary = {
        "tag": args.tag,
        "target": RV_INNOVATION_TARGET,
        "selection": "validation innovation RMSE only",
        "reservoir_policy": (
            "per-fold reservoir frozen from prior "
            "PCA6 validation search"
        ),
        "reservoir_config": str(
            args.reservoir_config
        ),
        "representation_grid": dims,
        "alphas": alphas,
        "median_test_metrics": (
            metrics.groupby("model")[
                [
                    "test_innovation_rmse",
                    "test_innovation_r2",
                    "test_reconstructed_rmse",
                ]
            ]
            .median()
            .to_dict(orient="index")
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
