from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.numpy_esn import esn_states, make_esn_weights
from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.modeling.stage_e_fixed_spec_diagnostics import (
    horizon_mz_summary,
    mincer_zarnowitz,
)
from transition_forecasting.modeling.stage_e_sequence_models import (
    DEFAULT_ESN_CONFIG,
    har_predictions,
    load_rematched_rolling,
    metrics,
    scale_sequences,
)


def _diagnostics(
    *,
    fold: int,
    model: str,
    representation: str,
    order: str,
    seed: int,
    alpha: float | None,
    train_samples: int,
    val_samples: int,
    y: np.ndarray,
    prediction: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, object]:
    qlike, rmse = metrics(y, prediction, val_mask)
    val_y = y[val_mask]
    val_prediction = prediction[val_mask]
    return {
        "fold": int(fold),
        "model": model,
        "representation": representation,
        "order": order,
        "seed": int(seed),
        "alpha": np.nan if alpha is None else float(alpha),
        "train_samples": int(train_samples),
        "val_samples": int(val_samples),
        "val_qlike": qlike,
        "val_rmse": rmse,
        **mincer_zarnowitz(val_y, val_prediction),
        **horizon_mz_summary(val_y, val_prediction),
    }


def _ridge_predictions(
    x_train: np.ndarray,
    x_all: np.ndarray,
    residual: np.ndarray,
    train_mask: np.ndarray,
    har: np.ndarray,
    alphas: tuple[float, ...],
) -> list[tuple[float, np.ndarray]]:
    predictions: list[tuple[float, np.ndarray]] = []
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(x_train, residual[train_mask])
        predictions.append((float(alpha), har + model.predict(x_all)))
    return predictions


def _scaled_states(states: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train = scaler.fit_transform(states[train_mask])
    all_rows = scaler.transform(states)
    return train, all_rows


def _evaluate_state_family(
    *,
    fold: int,
    order: str,
    seed: int,
    states: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    y: np.ndarray,
    har: np.ndarray,
    residual: np.ndarray,
    alphas: tuple[float, ...],
    pca_components: tuple[int, ...],
    pls_components: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    state_train, state_all = _scaled_states(states, train_mask)
    common = {
        "fold": fold,
        "order": order,
        "seed": seed,
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "y": y,
        "val_mask": val_mask,
    }

    for alpha, prediction in _ridge_predictions(
        state_train,
        state_all,
        residual,
        train_mask,
        har,
        alphas,
    ):
        rows.append(
            _diagnostics(
                model=f"{order}_raw_ridge",
                representation="raw",
                alpha=alpha,
                prediction=prediction,
                **common,
            )
        )

    for requested in pca_components:
        count = min(int(requested), state_train.shape[0], state_train.shape[1])
        if count < 1:
            continue
        pca = PCA(n_components=count, svd_solver="full")
        pca_train = pca.fit_transform(state_train)
        pca_all = pca.transform(state_all)
        for alpha, prediction in _ridge_predictions(
            pca_train,
            pca_all,
            residual,
            train_mask,
            har,
            alphas,
        ):
            rows.append(
                _diagnostics(
                    model=f"{order}_pca{count}_ridge",
                    representation=f"pca{count}",
                    alpha=alpha,
                    prediction=prediction,
                    **common,
                )
            )

    for requested in pls_components:
        count = min(
            int(requested),
            state_train.shape[0] - 1,
            state_train.shape[1],
            residual.shape[1],
        )
        if count < 1:
            continue
        pls = PLSRegression(n_components=count, scale=False, max_iter=1000)
        pls.fit(state_train, residual[train_mask])
        prediction = har + pls.predict(state_all)
        rows.append(
            _diagnostics(
                model=f"{order}_pls{count}",
                representation=f"pls{count}",
                alpha=None,
                prediction=prediction,
                **common,
            )
        )
    return rows


def _aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "val_qlike",
        "val_rmse",
        "mz_intercept",
        "mz_slope",
        "mz_r2",
        "mz_horizon_mean_abs_slope_error",
    ]
    seed_aggregations = {column: (column, "mean") for column in metric_columns}
    seed_aggregations.update(
        {
            "val_samples": ("val_samples", "first"),
            "train_samples": ("train_samples", "first"),
            "seeds": ("seed", "nunique"),
        }
    )
    per_fold = (
        raw.groupby(
            ["fold", "model", "representation", "order", "alpha"],
            as_index=False,
            dropna=False,
        )
        .agg(**seed_aggregations)
    )

    rows: list[dict[str, object]] = []
    for key, group in per_fold.groupby(
        ["model", "representation", "order", "alpha"],
        sort=True,
        dropna=False,
    ):
        model, representation, order, alpha = key
        weight = group["val_samples"].astype(float)
        rows.append(
            {
                "model": model,
                "representation": representation,
                "order": order,
                "alpha": alpha,
                "folds": int(group["fold"].nunique()),
                "weighted_qlike": float((group["val_qlike"] * weight).sum() / weight.sum()),
                "weighted_rmse": float((group["val_rmse"] * weight).sum() / weight.sum()),
                "weighted_mz_intercept": float((group["mz_intercept"] * weight).sum() / weight.sum()),
                "weighted_mz_slope": float((group["mz_slope"] * weight).sum() / weight.sum()),
                "weighted_mz_r2": float((group["mz_r2"] * weight).sum() / weight.sum()),
                "weighted_mz_horizon_mean_abs_slope_error": float(
                    (group["mz_horizon_mean_abs_slope_error"] * weight).sum() / weight.sum()
                ),
                "qlike_wins_vs_har": int((group["delta_qlike_vs_har"] < 0.0).sum())
                if "delta_qlike_vs_har" in group
                else 0,
            }
        )
    pooled = pd.DataFrame(rows).sort_values(["weighted_qlike", "weighted_rmse"]).reset_index(drop=True)
    return per_fold, pooled


def _attach_har_deltas(per_fold: pd.DataFrame) -> pd.DataFrame:
    har = per_fold[per_fold["model"].eq("har")][["fold", "val_qlike", "val_rmse"]].rename(
        columns={"val_qlike": "har_qlike", "val_rmse": "har_rmse"}
    )
    merged = per_fold.merge(har, on="fold", how="left")
    merged["delta_qlike_vs_har"] = merged["val_qlike"] - merged["har_qlike"]
    merged["delta_rmse_vs_har"] = merged["val_rmse"] - merged["har_rmse"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen ordered and shuffled ESN representations on development folds only."
    )
    parser.add_argument("--chronology-run", type=Path, required=True)
    parser.add_argument("--development-folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--alphas", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    parser.add_argument("--pca-components", type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--pls-components", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/transition_forecasting/modeling/run_stage_e_representation_screen"
        ),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, tensors = load_rematched_rolling(args.chronology_run)
    folds = tuple(int(fold) for fold in args.development_folds)
    seeds = tuple(int(seed) for seed in args.seeds)
    alphas = tuple(float(alpha) for alpha in args.alphas)
    pca_components = tuple(int(value) for value in args.pca_components)
    pls_components = tuple(int(value) for value in args.pls_components)
    config = dict(DEFAULT_ESN_CONFIG)

    rows: list[dict[str, object]] = []
    for fold in folds:
        mask = manifest["fold"].eq(fold) & manifest["fold_split"].isin(["train", "val"])
        frame = manifest.loc[mask].reset_index(drop=True)
        sequences = tensors[mask.to_numpy()]
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        residual = y - har
        rows.append(
            _diagnostics(
                fold=fold,
                model="har",
                representation="har",
                order="baseline",
                seed=0,
                alpha=100.0,
                train_samples=int(train_mask.sum()),
                val_samples=int(val_mask.sum()),
                y=y,
                prediction=har,
                val_mask=val_mask,
            )
        )

        scaled_sequences = scale_sequences(sequences, train_mask)
        for seed in seeds:
            w_in, w = make_esn_weights(
                n_inputs=scaled_sequences.shape[-1],
                n_reservoir=int(config["n"]),
                spectral_radius=float(config["sr"]),
                input_scale=float(config["inp"]),
                seed=seed,
            )
            ordered = esn_states(scaled_sequences, w_in, w, float(config["leak"]))
            rng = np.random.default_rng(seed + 20000)
            shuffled_sequences = scaled_sequences.copy()
            for sample in range(len(shuffled_sequences)):
                shuffled_sequences[sample] = shuffled_sequences[
                    sample, rng.permutation(shuffled_sequences.shape[1]), :
                ]
            shuffled = esn_states(shuffled_sequences, w_in, w, float(config["leak"]))

            rows.extend(
                _evaluate_state_family(
                    fold=fold,
                    order="ordered",
                    seed=seed,
                    states=ordered,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    y=y,
                    har=har,
                    residual=residual,
                    alphas=alphas,
                    pca_components=pca_components,
                    pls_components=pls_components,
                )
            )
            rows.extend(
                _evaluate_state_family(
                    fold=fold,
                    order="shuffled",
                    seed=seed,
                    states=shuffled,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    y=y,
                    har=har,
                    residual=residual,
                    alphas=alphas,
                    pca_components=pca_components,
                    pls_components=pls_components,
                )
            )

    raw = pd.DataFrame(rows)
    per_fold, _ = _aggregate(raw)
    per_fold = _attach_har_deltas(per_fold)

    pooled_rows: list[dict[str, object]] = []
    for key, group in per_fold.groupby(
        ["model", "representation", "order", "alpha"],
        sort=True,
        dropna=False,
    ):
        model, representation, order, alpha = key
        weight = group["val_samples"].astype(float)
        pooled_rows.append(
            {
                "model": model,
                "representation": representation,
                "order": order,
                "alpha": alpha,
                "folds": int(group["fold"].nunique()),
                "weighted_qlike": float((group["val_qlike"] * weight).sum() / weight.sum()),
                "weighted_rmse": float((group["val_rmse"] * weight).sum() / weight.sum()),
                "weighted_mz_intercept": float((group["mz_intercept"] * weight).sum() / weight.sum()),
                "weighted_mz_slope": float((group["mz_slope"] * weight).sum() / weight.sum()),
                "weighted_mz_r2": float((group["mz_r2"] * weight).sum() / weight.sum()),
                "weighted_mz_horizon_mean_abs_slope_error": float(
                    (group["mz_horizon_mean_abs_slope_error"] * weight).sum() / weight.sum()
                ),
                "mean_delta_qlike_vs_har": float(group["delta_qlike_vs_har"].mean()),
                "mean_delta_rmse_vs_har": float(group["delta_rmse_vs_har"].mean()),
                "qlike_wins_vs_har": int((group["delta_qlike_vs_har"] < 0.0).sum()),
                "rmse_wins_vs_har": int((group["delta_rmse_vs_har"] < 0.0).sum()),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values(
        ["weighted_qlike", "weighted_rmse", "weighted_mz_horizon_mean_abs_slope_error"]
    ).reset_index(drop=True)

    ordered = pooled[pooled["order"].eq("ordered")].copy()
    shuffled = pooled[pooled["order"].eq("shuffled")].copy()
    comparison = ordered.merge(
        shuffled,
        on=["representation", "alpha"],
        suffixes=("_ordered", "_shuffled"),
    )
    for metric in (
        "weighted_qlike",
        "weighted_rmse",
        "weighted_mz_r2",
        "weighted_mz_horizon_mean_abs_slope_error",
    ):
        comparison[f"ordered_minus_shuffled_{metric}"] = (
            comparison[f"{metric}_ordered"] - comparison[f"{metric}_shuffled"]
        )

    raw.to_csv(run_dir / "representation_seed_results.csv", index=False)
    per_fold.to_csv(run_dir / "representation_fold_results.csv", index=False)
    pooled.to_csv(run_dir / "representation_pooled_results.csv", index=False)
    comparison.to_csv(run_dir / "ordered_shuffled_comparison.csv", index=False)

    summary = {
        "chronology_run": str(args.chronology_run),
        "development_folds": list(folds),
        "confirmation_folds_untouched": [5, 6, 7, 8],
        "test_rows_used": 0,
        "reservoir_config": config,
        "seeds": list(seeds),
        "alphas": list(alphas),
        "pca_components": list(pca_components),
        "pls_components": list(pls_components),
        "selection_primary": "weighted QLIKE across development folds",
        "selection_secondary": [
            "ordered-minus-shuffled evidence",
            "weighted RMSE",
            "MZ calibration",
        ],
        "best_ordered_by_qlike": ordered.iloc[0][
            ["model", "representation", "alpha", "weighted_qlike", "weighted_rmse"]
        ].to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nTop ordered candidates:\n")
    print(
        ordered.head(15)[
            [
                "model",
                "representation",
                "alpha",
                "weighted_qlike",
                "weighted_rmse",
                "weighted_mz_slope",
                "weighted_mz_horizon_mean_abs_slope_error",
                "qlike_wins_vs_har",
                "rmse_wins_vs_har",
            ]
        ].to_string(index=False)
    )
    print("\nBest ordered-minus-shuffled comparisons by QLIKE:\n")
    print(
        comparison.sort_values("ordered_minus_shuffled_weighted_qlike").head(15)[
            [
                "representation",
                "alpha",
                "ordered_minus_shuffled_weighted_qlike",
                "ordered_minus_shuffled_weighted_rmse",
                "ordered_minus_shuffled_weighted_mz_r2",
                "ordered_minus_shuffled_weighted_mz_horizon_mean_abs_slope_error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
