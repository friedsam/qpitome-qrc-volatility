from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
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


def evaluate_prediction(
    *,
    fold: int,
    order: str,
    seed: int,
    components: int,
    shrinkage: float,
    y: np.ndarray,
    prediction: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, object]:
    qlike, rmse = metrics(y, prediction, val_mask)
    val_y = y[val_mask]
    val_prediction = prediction[val_mask]
    return {
        "fold": int(fold),
        "order": order,
        "seed": int(seed),
        "components": int(components),
        "shrinkage": float(shrinkage),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "val_qlike": qlike,
        "val_rmse": rmse,
        **mincer_zarnowitz(val_y, val_prediction),
        **horizon_mz_summary(val_y, val_prediction),
    }


def aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "val_qlike",
        "val_rmse",
        "mz_intercept",
        "mz_slope",
        "mz_r2",
        "mz_horizon_mean_abs_slope_error",
    ]
    per_fold = (
        raw.groupby(["fold", "order", "components", "shrinkage"], as_index=False)
        .agg(
            **{column: (column, "mean") for column in metric_columns},
            val_samples=("val_samples", "first"),
            train_samples=("train_samples", "first"),
            seeds=("seed", "nunique"),
        )
    )

    pooled_rows: list[dict[str, object]] = []
    for key, group in per_fold.groupby(["order", "components", "shrinkage"], sort=True):
        order, components, shrinkage = key
        weight = group["val_samples"].astype(float)
        pooled_rows.append(
            {
                "order": order,
                "components": int(components),
                "shrinkage": float(shrinkage),
                "folds": int(group["fold"].nunique()),
                "weighted_qlike": float((group["val_qlike"] * weight).sum() / weight.sum()),
                "weighted_rmse": float((group["val_rmse"] * weight).sum() / weight.sum()),
                "weighted_mz_intercept": float((group["mz_intercept"] * weight).sum() / weight.sum()),
                "weighted_mz_slope": float((group["mz_slope"] * weight).sum() / weight.sum()),
                "weighted_mz_r2": float((group["mz_r2"] * weight).sum() / weight.sum()),
                "weighted_mz_horizon_mean_abs_slope_error": float(
                    (group["mz_horizon_mean_abs_slope_error"] * weight).sum() / weight.sum()
                ),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values(["weighted_qlike", "weighted_rmse"]).reset_index(drop=True)

    ordered = pooled[pooled["order"].eq("ordered")]
    shuffled = pooled[pooled["order"].eq("shuffled")]
    comparison = ordered.merge(
        shuffled,
        on=["components", "shrinkage"],
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
    return per_fold, pooled, comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose PLS dimensionality and residual shrinkage on development folds."
    )
    parser.add_argument("--chronology-run", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument(
        "--shrinkages",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/run_stage_e_pls_diagnostics"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, tensors = load_rematched_rolling(args.chronology_run)
    config = dict(DEFAULT_ESN_CONFIG)
    rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []

    for fold in tuple(int(value) for value in args.folds):
        mask = manifest["fold"].eq(fold) & manifest["fold_split"].isin(["train", "val"])
        frame = manifest.loc[mask].reset_index(drop=True)
        sequences = tensors[mask.to_numpy()]
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        residual = y - har
        scaled = scale_sequences(sequences, train_mask)

        for seed in tuple(int(value) for value in args.seeds):
            w_in, w = make_esn_weights(
                n_inputs=scaled.shape[-1],
                n_reservoir=int(config["n"]),
                spectral_radius=float(config["sr"]),
                input_scale=float(config["inp"]),
                seed=seed,
            )
            ordered_states = esn_states(scaled, w_in, w, float(config["leak"]))
            rng = np.random.default_rng(seed + 20000)
            shuffled_sequences = scaled.copy()
            for sample in range(len(shuffled_sequences)):
                shuffled_sequences[sample] = shuffled_sequences[
                    sample, rng.permutation(shuffled_sequences.shape[1]), :
                ]
            shuffled_states = esn_states(shuffled_sequences, w_in, w, float(config["leak"]))

            for order, states in (("ordered", ordered_states), ("shuffled", shuffled_states)):
                scaler = StandardScaler()
                state_train = scaler.fit_transform(states[train_mask])
                state_all = scaler.transform(states)
                max_components = min(
                    int(args.max_components),
                    state_train.shape[0] - 1,
                    state_train.shape[1],
                    residual.shape[1],
                )
                for components in range(1, max_components + 1):
                    pls = PLSRegression(n_components=components, scale=False, max_iter=1000)
                    pls.fit(state_train, residual[train_mask])
                    correction = pls.predict(state_all)
                    for shrinkage in tuple(float(value) for value in args.shrinkages):
                        prediction = har + shrinkage * correction
                        rows.append(
                            evaluate_prediction(
                                fold=fold,
                                order=order,
                                seed=seed,
                                components=components,
                                shrinkage=shrinkage,
                                y=y,
                                prediction=prediction,
                                train_mask=train_mask,
                                val_mask=val_mask,
                            )
                        )
                    if order == "ordered" and components == max_components:
                        weights = np.asarray(pls.x_weights_, dtype=float)
                        for component in range(weights.shape[1]):
                            absolute = np.abs(weights[:, component])
                            top = np.argsort(absolute)[-10:][::-1]
                            for rank, feature in enumerate(top, start=1):
                                loading_rows.append(
                                    {
                                        "fold": fold,
                                        "seed": seed,
                                        "component": component + 1,
                                        "rank": rank,
                                        "reservoir_feature": int(feature),
                                        "weight": float(weights[feature, component]),
                                        "abs_weight": float(absolute[feature]),
                                    }
                                )

    raw = pd.DataFrame(rows)
    per_fold, pooled, comparison = aggregate(raw)
    loadings = pd.DataFrame(loading_rows)

    raw.to_csv(run_dir / "pls_seed_results.csv", index=False)
    per_fold.to_csv(run_dir / "pls_fold_results.csv", index=False)
    pooled.to_csv(run_dir / "pls_pooled_results.csv", index=False)
    comparison.to_csv(run_dir / "pls_ordered_shuffled.csv", index=False)
    loadings.to_csv(run_dir / "pls_top_loadings.csv", index=False)

    ordered = pooled[pooled["order"].eq("ordered")]
    summary = {
        "chronology_run": str(args.chronology_run),
        "development_folds": [int(value) for value in args.folds],
        "test_rows_used": 0,
        "reservoir_config": config,
        "max_components": int(args.max_components),
        "shrinkages": [float(value) for value in args.shrinkages],
        "best_ordered_by_qlike": ordered.iloc[0][
            [
                "components",
                "shrinkage",
                "weighted_qlike",
                "weighted_rmse",
                "weighted_mz_slope",
                "weighted_mz_r2",
            ]
        ].to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nTop ordered component/shrinkage combinations:\n")
    print(
        ordered.head(20)[
            [
                "components",
                "shrinkage",
                "weighted_qlike",
                "weighted_rmse",
                "weighted_mz_slope",
                "weighted_mz_r2",
                "weighted_mz_horizon_mean_abs_slope_error",
            ]
        ].to_string(index=False)
    )
    print("\nBest ordered-minus-shuffled combinations:\n")
    print(
        comparison.sort_values("ordered_minus_shuffled_weighted_qlike").head(20)[
            [
                "components",
                "shrinkage",
                "ordered_minus_shuffled_weighted_qlike",
                "ordered_minus_shuffled_weighted_rmse",
                "ordered_minus_shuffled_weighted_mz_r2",
                "ordered_minus_shuffled_weighted_mz_horizon_mean_abs_slope_error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
