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
from transition_forecasting.modeling.stage_e_classical_baselines import (
    TARGET_COLUMNS,
    qlike_loss,
)
from transition_forecasting.modeling.stage_e_sequence_models import (
    DEFAULT_ESN_CONFIG,
    har_predictions,
    load_rematched_rolling,
    scale_sequences,
)

FROZEN_SPECS = {
    "pls5_s0.5": (5, 0.5),
    "pls10_s0.75": (10, 0.75),
}


def _shuffle_sequences(sequences: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 20000)
    shuffled = sequences.copy()
    for sample in range(len(shuffled)):
        shuffled[sample] = shuffled[sample, rng.permutation(shuffled.shape[1]), :]
    return shuffled


def _fit_pls_predictions(
    *,
    states: np.ndarray,
    residual: np.ndarray,
    har: np.ndarray,
    train_mask: np.ndarray,
    max_components: int,
    shrinkages: tuple[float, ...],
) -> dict[tuple[int, float], np.ndarray]:
    scaler = StandardScaler()
    state_train = scaler.fit_transform(states[train_mask])
    state_all = scaler.transform(states)
    available = min(
        int(max_components),
        state_train.shape[0] - 1,
        state_train.shape[1],
        residual.shape[1],
    )
    predictions: dict[tuple[int, float], np.ndarray] = {}
    for components in range(1, available + 1):
        pls = PLSRegression(n_components=components, scale=False, max_iter=1000)
        pls.fit(state_train, residual[train_mask])
        correction = pls.predict(state_all)
        for shrinkage in shrinkages:
            predictions[(components, shrinkage)] = har + float(shrinkage) * correction
    return predictions


def _prediction_rows(
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    val_mask: np.ndarray,
    fold: int,
    seed: int,
    model: str,
    order: str,
    components: int | None,
    shrinkage: float | None,
) -> list[dict[str, object]]:
    val_frame = frame.loc[val_mask].reset_index(drop=True)
    val_y = y[val_mask]
    val_prediction = prediction[val_mask]
    losses = qlike_loss(val_y, val_prediction)
    errors = val_y - val_prediction
    rows: list[dict[str, object]] = []
    metadata_columns = [
        column
        for column in (
            "sample_id",
            "index",
            "market_group",
            "episode_id",
            "label",
            "lead",
            "origin_date",
            "event_onset",
        )
        if column in val_frame.columns
    ]
    for sample_position in range(len(val_frame)):
        metadata = val_frame.loc[sample_position, metadata_columns].to_dict()
        for horizon in range(val_y.shape[1]):
            rows.append(
                {
                    **metadata,
                    "fold": int(fold),
                    "seed": int(seed),
                    "model": model,
                    "order": order,
                    "components": np.nan if components is None else int(components),
                    "shrinkage": np.nan if shrinkage is None else float(shrinkage),
                    "horizon": horizon + 1,
                    "observed": float(val_y[sample_position, horizon]),
                    "prediction": float(val_prediction[sample_position, horizon]),
                    "forecast_error": float(errors[sample_position, horizon]),
                    "absolute_error": float(abs(errors[sample_position, horizon])),
                    "squared_error": float(errors[sample_position, horizon] ** 2),
                    "qlike_contribution": float(losses[sample_position, horizon]),
                    "underprediction": bool(errors[sample_position, horizon] > 0.0),
                }
            )
    return rows


def _metric_summary(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in predictions.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key, strict=True))
        row.update(
            n_points=int(len(group)),
            n_samples=int(group["sample_id"].nunique()) if "sample_id" in group else 0,
            rmse=float(np.sqrt(group["squared_error"].mean())),
            qlike=float(group["qlike_contribution"].mean()),
            mean_error=float(group["forecast_error"].mean()),
            median_error=float(group["forecast_error"].median()),
            underprediction_rate=float(group["underprediction"].mean()),
            qlike_p95=float(group["qlike_contribution"].quantile(0.95)),
            qlike_p99=float(group["qlike_contribution"].quantile(0.99)),
            max_qlike=float(group["qlike_contribution"].max()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_differences(summary: pd.DataFrame, pair_column: str, first: str, second: str) -> pd.DataFrame:
    key_columns = [
        column
        for column in summary.columns
        if column
        not in {
            pair_column,
            "n_points",
            "n_samples",
            "rmse",
            "qlike",
            "mean_error",
            "median_error",
            "underprediction_rate",
            "qlike_p95",
            "qlike_p99",
            "max_qlike",
        }
    ]
    metric_columns = [
        "rmse",
        "qlike",
        "mean_error",
        "underprediction_rate",
        "qlike_p95",
        "qlike_p99",
        "max_qlike",
    ]
    left = summary[summary[pair_column].eq(first)]
    right = summary[summary[pair_column].eq(second)]
    merged = left.merge(right, on=key_columns, suffixes=(f"_{first}", f"_{second}"), validate="one_to_one")
    for metric in metric_columns:
        merged[f"{first}_minus_{second}_{metric}"] = merged[f"{metric}_{first}"] - merged[f"{metric}_{second}"]
    return merged


def _component_summary(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pls = predictions[predictions["model"].eq("pls_path")].copy()
    per_fold_seed = _metric_summary(
        pls,
        ["fold", "seed", "order", "components", "shrinkage"],
    )
    per_fold = (
        per_fold_seed.groupby(["fold", "order", "components", "shrinkage"], as_index=False)
        .agg(
            val_samples=("n_samples", "first"),
            qlike=("qlike", "mean"),
            rmse=("rmse", "mean"),
            mean_error=("mean_error", "mean"),
            qlike_p99=("qlike_p99", "mean"),
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
                "weighted_qlike": float((group["qlike"] * weight).sum() / weight.sum()),
                "weighted_rmse": float((group["rmse"] * weight).sum() / weight.sum()),
                "weighted_mean_error": float((group["mean_error"] * weight).sum() / weight.sum()),
                "weighted_qlike_p99": float((group["qlike_p99"] * weight).sum() / weight.sum()),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    ordered = pooled[pooled["order"].eq("ordered")]
    shuffled = pooled[pooled["order"].eq("shuffled")]
    comparison = ordered.merge(
        shuffled,
        on=["components", "shrinkage"],
        suffixes=("_ordered", "_shuffled"),
        validate="one_to_one",
    )
    for metric in ("weighted_qlike", "weighted_rmse", "weighted_mean_error", "weighted_qlike_p99"):
        comparison[f"ordered_minus_shuffled_{metric}"] = comparison[f"{metric}_ordered"] - comparison[f"{metric}_shuffled"]
    comparison = comparison.sort_values(["shrinkage", "components"]).reset_index(drop=True)
    comparison["incremental_order_advantage_qlike"] = comparison.groupby("shrinkage")[
        "ordered_minus_shuffled_weighted_qlike"
    ].diff()
    comparison["incremental_order_advantage_rmse"] = comparison.groupby("shrinkage")[
        "ordered_minus_shuffled_weighted_rmse"
    ].diff()
    return per_fold, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact conditional, component, and QLIKE-tail diagnostics for frozen Stage E models.")
    parser.add_argument("--chronology-run", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument("--component-shrinkages", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--out-dir", type=Path, default=Path("results/transition_forecasting/modeling/run_stage_e_compact_diagnostics"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, tensors = load_rematched_rolling(args.chronology_run)
    config = dict(DEFAULT_ESN_CONFIG)
    prediction_rows: list[dict[str, object]] = []

    for fold in (int(value) for value in args.folds):
        mask = manifest["fold"].eq(fold) & manifest["fold_split"].isin(["train", "val"])
        frame = manifest.loc[mask].reset_index(drop=True)
        sequences = tensors[mask.to_numpy()]
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        residual = y - har
        scaled = scale_sequences(sequences, train_mask)

        prediction_rows.extend(
            _prediction_rows(
                frame=frame,
                y=y,
                prediction=har,
                val_mask=val_mask,
                fold=fold,
                seed=0,
                model="har",
                order="baseline",
                components=None,
                shrinkage=None,
            )
        )

        for seed in (int(value) for value in args.seeds):
            w_in, w = make_esn_weights(
                n_inputs=scaled.shape[-1],
                n_reservoir=int(config["n"]),
                spectral_radius=float(config["sr"]),
                input_scale=float(config["inp"]),
                seed=seed,
            )
            ordered_states = esn_states(scaled, w_in, w, float(config["leak"]))
            shuffled_states = esn_states(_shuffle_sequences(scaled, seed), w_in, w, float(config["leak"]))

            for order, states in (("ordered", ordered_states), ("shuffled", shuffled_states)):
                predictions = _fit_pls_predictions(
                    states=states,
                    residual=residual,
                    har=har,
                    train_mask=train_mask,
                    max_components=int(args.max_components),
                    shrinkages=tuple(float(value) for value in args.component_shrinkages),
                )
                for (components, shrinkage), prediction in predictions.items():
                    prediction_rows.extend(
                        _prediction_rows(
                            frame=frame,
                            y=y,
                            prediction=prediction,
                            val_mask=val_mask,
                            fold=fold,
                            seed=seed,
                            model="pls_path",
                            order=order,
                            components=components,
                            shrinkage=shrinkage,
                        )
                    )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(run_dir / "validation_point_predictions.csv.gz", index=False, compression="gzip")

    frozen_masks = [predictions["model"].eq("har")]
    for _, (components, shrinkage) in FROZEN_SPECS.items():
        frozen_masks.append(
            predictions["model"].eq("pls_path")
            & predictions["components"].eq(components)
            & np.isclose(predictions["shrinkage"], shrinkage)
        )
    frozen = predictions[np.logical_or.reduce(frozen_masks)].copy()
    frozen["frozen_model"] = "har"
    for name, (components, shrinkage) in FROZEN_SPECS.items():
        mask = (
            frozen["model"].eq("pls_path")
            & frozen["components"].eq(components)
            & np.isclose(frozen["shrinkage"], shrinkage)
        )
        frozen.loc[mask, "frozen_model"] = name

    group_specs = {
        "lead_horizon": ["frozen_model", "order", "lead", "horizon"],
        "label_horizon": ["frozen_model", "order", "label", "horizon"],
        "lead_label_horizon": ["frozen_model", "order", "lead", "label", "horizon"],
        "fold": ["frozen_model", "order", "fold"],
        "fold_lead_label": ["frozen_model", "order", "fold", "lead", "label"],
    }
    summaries: dict[str, pd.DataFrame] = {}
    for name, columns in group_specs.items():
        summary = _metric_summary(frozen, columns)
        summary.to_csv(run_dir / f"{name}_metrics.csv", index=False)
        summaries[name] = summary

    paired_source = summaries["lead_label_horizon"]
    for frozen_model in FROZEN_SPECS:
        subset = paired_source[paired_source["frozen_model"].eq(frozen_model)]
        paired = _paired_differences(subset, "order", "ordered", "shuffled")
        paired.to_csv(run_dir / f"{frozen_model}_ordered_minus_shuffled_by_lead_label_horizon.csv", index=False)

    component_fold, component_comparison = _component_summary(predictions)
    component_fold.to_csv(run_dir / "component_fold_metrics.csv", index=False)
    component_comparison.to_csv(run_dir / "component_order_signal.csv", index=False)

    tail_columns = [
        column
        for column in (
            "fold",
            "frozen_model",
            "order",
            "seed",
            "sample_id",
            "index",
            "market_group",
            "episode_id",
            "label",
            "lead",
            "origin_date",
            "event_onset",
            "horizon",
            "observed",
            "prediction",
            "forecast_error",
            "underprediction",
            "qlike_contribution",
        )
        if column in frozen.columns
    ]
    tails = frozen.sort_values("qlike_contribution", ascending=False).head(1000)[tail_columns]
    tails.to_csv(run_dir / "top_1000_qlike_contributions.csv", index=False)
    fold6 = frozen[frozen["fold"].eq(6)].sort_values("qlike_contribution", ascending=False)
    fold6.head(1000)[tail_columns].to_csv(run_dir / "fold6_top_1000_qlike_contributions.csv", index=False)

    tail_concentration_rows: list[dict[str, object]] = []
    for key, group in frozen.groupby(["fold", "frozen_model", "order"], sort=True):
        fold, frozen_model, order = key
        sorted_loss = np.sort(group["qlike_contribution"].to_numpy(dtype=float))[::-1]
        total = float(sorted_loss.sum())
        row: dict[str, object] = {
            "fold": int(fold),
            "frozen_model": frozen_model,
            "order": order,
            "n_points": int(len(sorted_loss)),
            "mean_qlike": float(sorted_loss.mean()),
        }
        for fraction in (0.001, 0.01, 0.05, 0.10):
            count = max(1, int(np.ceil(fraction * len(sorted_loss))))
            row[f"top_{fraction:g}_fraction_of_total_qlike"] = float(sorted_loss[:count].sum() / total) if total > 0.0 else 0.0
        tail_concentration_rows.append(row)
    tail_concentration = pd.DataFrame(tail_concentration_rows)
    tail_concentration.to_csv(run_dir / "qlike_tail_concentration.csv", index=False)

    summary = {
        "chronology_run": str(args.chronology_run),
        "folds": [int(value) for value in args.folds],
        "seeds": [int(value) for value in args.seeds],
        "reservoir_config": config,
        "frozen_specs": {name: {"components": c, "shrinkage": s} for name, (c, s) in FROZEN_SPECS.items()},
        "component_path": {"max_components": int(args.max_components), "shrinkages": [float(value) for value in args.component_shrinkages]},
        "test_rows_used": 0,
        "outputs": [
            "lead/horizon metrics",
            "transition/control metrics",
            "lead-label-horizon ordered/shuffled comparisons",
            "fold and fold-6 QLIKE tail diagnostics",
            "PLS component order-signal path",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nFrozen fold metrics:\n")
    print(summaries["fold"].to_string(index=False))
    print("\nPLS component order signal:\n")
    print(
        component_comparison[
            [
                "components",
                "shrinkage",
                "ordered_minus_shuffled_weighted_qlike",
                "ordered_minus_shuffled_weighted_rmse",
                "incremental_order_advantage_qlike",
                "incremental_order_advantage_rmse",
            ]
        ].to_string(index=False)
    )
    print("\nQLIKE tail concentration:\n")
    print(tail_concentration.to_string(index=False))


if __name__ == "__main__":
    main()
