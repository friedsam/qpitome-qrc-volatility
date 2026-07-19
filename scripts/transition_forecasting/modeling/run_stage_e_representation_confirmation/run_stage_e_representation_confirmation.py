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

FROZEN_CANDIDATES = (
    {"name": "pls5", "kind": "pls", "components": 5, "alpha": None},
    {"name": "pls10", "kind": "pls", "components": 10, "alpha": None},
    {"name": "pca5_a10000", "kind": "pca_ridge", "components": 5, "alpha": 10000.0},
)


def diagnostics(
    fold: int,
    model: str,
    candidate: str,
    order: str,
    seed: int,
    y: np.ndarray,
    prediction: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, object]:
    qlike, rmse = metrics(y, prediction, val_mask)
    val_y = y[val_mask]
    val_prediction = prediction[val_mask]
    return {
        "fold": fold,
        "model": model,
        "candidate": candidate,
        "order": order,
        "seed": seed,
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "val_qlike": qlike,
        "val_rmse": rmse,
        **mincer_zarnowitz(val_y, val_prediction),
        **horizon_mz_summary(val_y, val_prediction),
    }


def transform_states(
    states: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(states[train_mask]), scaler.transform(states)


def candidate_prediction(
    spec: dict[str, object],
    state_train: np.ndarray,
    state_all: np.ndarray,
    residual: np.ndarray,
    train_mask: np.ndarray,
    har: np.ndarray,
) -> np.ndarray:
    components = min(
        int(spec["components"]),
        state_train.shape[0] - 1,
        state_train.shape[1],
    )
    if spec["kind"] == "pls":
        components = min(components, residual.shape[1])
        model = PLSRegression(n_components=components, scale=False, max_iter=1000)
        model.fit(state_train, residual[train_mask])
        return har + model.predict(state_all)
    if spec["kind"] == "pca_ridge":
        pca = PCA(n_components=components, svd_solver="full")
        x_train = pca.fit_transform(state_train)
        x_all = pca.transform(state_all)
        model = Ridge(alpha=float(spec["alpha"]))
        model.fit(x_train, residual[train_mask])
        return har + model.predict(x_all)
    raise ValueError(f"unknown candidate kind: {spec['kind']}")


def aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_columns = [
        "val_qlike",
        "val_rmse",
        "mz_intercept",
        "mz_slope",
        "mz_r2",
        "mz_horizon_mean_abs_slope_error",
    ]
    aggregations = {column: (column, "mean") for column in metrics_columns}
    aggregations.update(
        {
            "train_samples": ("train_samples", "first"),
            "val_samples": ("val_samples", "first"),
            "seeds": ("seed", "nunique"),
        }
    )
    per_fold = raw.groupby(
        ["fold", "model", "candidate", "order"], as_index=False
    ).agg(**aggregations)

    har = per_fold[per_fold["model"].eq("har")][
        ["fold", "val_qlike", "val_rmse"]
    ].rename(columns={"val_qlike": "har_qlike", "val_rmse": "har_rmse"})
    per_fold = per_fold.merge(har, on="fold", how="left")
    per_fold["delta_qlike_vs_har"] = per_fold["val_qlike"] - per_fold["har_qlike"]
    per_fold["delta_rmse_vs_har"] = per_fold["val_rmse"] - per_fold["har_rmse"]

    pooled_rows: list[dict[str, object]] = []
    for key, group in per_fold.groupby(["model", "candidate", "order"], sort=True):
        model, candidate, order = key
        weight = group["val_samples"].astype(float)
        pooled_rows.append(
            {
                "model": model,
                "candidate": candidate,
                "order": order,
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
                "qlike_wins_vs_har": int((group["delta_qlike_vs_har"] < 0).sum()),
                "rmse_wins_vs_har": int((group["delta_rmse_vs_har"] < 0).sum()),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values("weighted_qlike").reset_index(drop=True)

    ordered = pooled[pooled["order"].eq("ordered")]
    shuffled = pooled[pooled["order"].eq("shuffled")]
    comparison = ordered.merge(shuffled, on="candidate", suffixes=("_ordered", "_shuffled"))
    for column in (
        "weighted_qlike",
        "weighted_rmse",
        "weighted_mz_r2",
        "weighted_mz_horizon_mean_abs_slope_error",
    ):
        comparison[f"ordered_minus_shuffled_{column}"] = (
            comparison[f"{column}_ordered"] - comparison[f"{column}_shuffled"]
        )
    return per_fold, pooled, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm frozen ESN representations on folds 5-8.")
    parser.add_argument("--chronology-run", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/run_stage_e_representation_confirmation"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, tensors = load_rematched_rolling(args.chronology_run)
    config = dict(DEFAULT_ESN_CONFIG)
    rows: list[dict[str, object]] = []

    for fold in tuple(int(value) for value in args.folds):
        mask = manifest["fold"].eq(fold) & manifest["fold_split"].isin(["train", "val"])
        frame = manifest.loc[mask].reset_index(drop=True)
        sequences = tensors[mask.to_numpy()]
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        residual = y - har
        rows.append(diagnostics(fold, "har", "har", "baseline", 0, y, har, train_mask, val_mask))

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
                state_train, state_all = transform_states(states, train_mask)
                for spec in FROZEN_CANDIDATES:
                    prediction = candidate_prediction(
                        spec, state_train, state_all, residual, train_mask, har
                    )
                    rows.append(
                        diagnostics(
                            fold,
                            f"{order}_{spec['name']}",
                            str(spec["name"]),
                            order,
                            seed,
                            y,
                            prediction,
                            train_mask,
                            val_mask,
                        )
                    )

    raw = pd.DataFrame(rows)
    per_fold, pooled, comparison = aggregate(raw)
    raw.to_csv(run_dir / "confirmation_seed_results.csv", index=False)
    per_fold.to_csv(run_dir / "confirmation_fold_results.csv", index=False)
    pooled.to_csv(run_dir / "confirmation_pooled_results.csv", index=False)
    comparison.to_csv(run_dir / "confirmation_ordered_shuffled.csv", index=False)

    summary = {
        "chronology_run": str(args.chronology_run),
        "confirmation_folds": [int(value) for value in args.folds],
        "frozen_candidates": list(FROZEN_CANDIDATES),
        "selection_origin": "development folds 1-4 only",
        "test_rows_used": 0,
        "reservoir_config": config,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nConfirmation pooled results:\n")
    print(pooled.to_string(index=False))
    print("\nOrdered minus shuffled:\n")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
