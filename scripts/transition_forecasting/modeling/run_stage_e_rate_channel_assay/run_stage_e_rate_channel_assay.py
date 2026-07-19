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
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS, qlike_loss
from transition_forecasting.modeling.stage_e_sequence_models import (
    DEFAULT_ESN_CONFIG,
    har_predictions,
    load_rematched_rolling,
    scale_sequences,
)

PLS_SPECS = ((5, 0.5), (10, 0.75))


def build_input_family(sequences: np.ndarray, family: str) -> np.ndarray:
    """Construct causal channels from the stored log-volatility-like sequence.

    The first finite difference is set to zero because the tensor does not include
    the observation immediately preceding its 40-row window. All channels are
    subsequently standardized from fold-training rows only.
    """
    x = np.asarray(sequences, dtype=float)
    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError(f"expected sequences shaped (n, time, 1), got {x.shape}")

    level = x[:, :, 0]
    relative_rate = np.diff(level, axis=1, prepend=level[:, :1])
    volatility = np.exp(np.clip(level, -20.0, 20.0))
    absolute_rate = np.diff(volatility, axis=1, prepend=volatility[:, :1])
    smooth_level = np.empty_like(level)
    smooth_level[:, 0] = level[:, 0]
    smooth_level[:, 1:] = 0.5 * (level[:, 1:] + level[:, :-1])
    smooth_relative_rate = np.diff(smooth_level, axis=1, prepend=smooth_level[:, :1])

    families = {
        "level": level[:, :, None],
        "relative_rate": relative_rate[:, :, None],
        "level_relative_rate": np.stack((level, relative_rate), axis=-1),
        "absolute_rate": absolute_rate[:, :, None],
        "level_absolute_rate": np.stack((level, absolute_rate), axis=-1),
        "level_smoothed_relative_rate": np.stack((level, smooth_relative_rate), axis=-1),
    }
    try:
        result = families[family]
    except KeyError as exc:
        raise ValueError(f"unknown input family {family!r}; choose from {sorted(families)}") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"input family {family!r} contains non-finite values")
    return result


def independently_shuffle_time(sequences: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 20000)
    result = np.asarray(sequences, dtype=float).copy()
    for sample in range(len(result)):
        result[sample] = result[sample, rng.permutation(result.shape[1]), :]
    return result


def fit_pls_hybrid(
    states: np.ndarray,
    residual: np.ndarray,
    har: np.ndarray,
    train_mask: np.ndarray,
    components: int,
    shrinkage: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_states = scaler.fit_transform(states[train_mask])
    all_states = scaler.transform(states)
    count = min(int(components), train_states.shape[0] - 1, train_states.shape[1], residual.shape[1])
    if count < 1:
        raise ValueError("insufficient rank for PLS fit")
    model = PLSRegression(n_components=count, scale=False, max_iter=1000)
    model.fit(train_states, residual[train_mask])
    return har + float(shrinkage) * model.predict(all_states)


def metric_row(
    *,
    y: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    fold: int,
    seed: int,
    family: str,
    order: str,
    components: int,
    shrinkage: float,
    lead: int | str,
    label: int | str,
) -> dict[str, object]:
    yt = y[mask]
    yp = prediction[mask]
    errors = yt - yp
    losses = qlike_loss(yt, yp)
    return {
        "fold": int(fold),
        "seed": int(seed),
        "input_family": family,
        "order": order,
        "components": int(components),
        "shrinkage": float(shrinkage),
        "lead": lead,
        "label": label,
        "n_samples": int(mask.sum()),
        "n_points": int(errors.size),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "qlike": float(losses.mean()),
        "mean_error": float(errors.mean()),
        "underprediction_rate": float((errors > 0.0).mean()),
        "qlike_p99": float(np.quantile(losses, 0.99)),
        "max_qlike": float(losses.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact assay of explicit volatility-rate channels in the frozen Stage E ESN.")
    parser.add_argument("--chronology-run", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--input-families",
        nargs="+",
        default=[
            "level",
            "relative_rate",
            "level_relative_rate",
            "absolute_rate",
            "level_absolute_rate",
            "level_smoothed_relative_rate",
        ],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/run_stage_e_rate_channel_assay"),
    )
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, tensors = load_rematched_rolling(args.chronology_run)
    config = dict(DEFAULT_ESN_CONFIG)
    rows: list[dict[str, object]] = []

    for fold in map(int, args.folds):
        fold_mask = manifest["fold"].eq(fold) & manifest["fold_split"].isin(["train", "val"])
        frame = manifest.loc[fold_mask].reset_index(drop=True)
        base_sequences = tensors[fold_mask.to_numpy()]
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        if not train_mask.any() or not val_mask.any():
            raise ValueError(f"fold {fold} has an empty training or validation split")
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        residual = y - har

        for family in args.input_families:
            transformed = build_input_family(base_sequences, family)
            scaled = scale_sequences(transformed, train_mask)
            for seed in map(int, args.seeds):
                w_in, w = make_esn_weights(
                    n_inputs=scaled.shape[-1],
                    n_reservoir=int(config["n"]),
                    spectral_radius=float(config["sr"]),
                    input_scale=float(config["inp"]),
                    seed=seed,
                )
                ordered_states = esn_states(scaled, w_in, w, float(config["leak"]))
                shuffled_states = esn_states(independently_shuffle_time(scaled, seed), w_in, w, float(config["leak"]))

                for order, states in (("ordered", ordered_states), ("shuffled", shuffled_states)):
                    for components, shrinkage in PLS_SPECS:
                        prediction = fit_pls_hybrid(
                            states,
                            residual,
                            har,
                            train_mask,
                            components,
                            shrinkage,
                        )
                        group_masks: list[tuple[int | str, int | str, np.ndarray]] = [("all", "all", val_mask)]
                        for lead in sorted(frame.loc[val_mask, "lead"].unique()):
                            lead_mask = val_mask & frame["lead"].eq(lead).to_numpy()
                            group_masks.append((int(lead), "all", lead_mask))
                            for label in (0, 1):
                                group_masks.append(
                                    (
                                        int(lead),
                                        int(label),
                                        lead_mask & frame["label"].eq(label).to_numpy(),
                                    )
                                )
                        for lead, label, group_mask in group_masks:
                            if group_mask.any():
                                rows.append(
                                    metric_row(
                                        y=y,
                                        prediction=prediction,
                                        mask=group_mask,
                                        fold=fold,
                                        seed=seed,
                                        family=family,
                                        order=order,
                                        components=components,
                                        shrinkage=shrinkage,
                                        lead=lead,
                                        label=label,
                                    )
                                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / "fold_seed_metrics.csv", index=False)

    grouping = ["input_family", "order", "components", "shrinkage", "lead", "label"]
    summary = (
        metrics.groupby(grouping, as_index=False, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            seeds=("seed", "nunique"),
            mean_qlike=("qlike", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_error=("mean_error", "mean"),
            underprediction_rate=("underprediction_rate", "mean"),
            mean_qlike_p99=("qlike_p99", "mean"),
            worst_max_qlike=("max_qlike", "max"),
        )
    )
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)

    ordered = summary[summary["order"].eq("ordered")]
    shuffled = summary[summary["order"].eq("shuffled")]
    keys = ["input_family", "components", "shrinkage", "lead", "label"]
    comparison = ordered.merge(shuffled, on=keys, suffixes=("_ordered", "_shuffled"), validate="one_to_one")
    for metric in ("mean_qlike", "mean_rmse", "mean_error", "mean_qlike_p99", "worst_max_qlike"):
        comparison[f"ordered_minus_shuffled_{metric}"] = comparison[f"{metric}_ordered"] - comparison[f"{metric}_shuffled"]
    comparison.to_csv(run_dir / "ordered_minus_shuffled.csv", index=False)

    focus = comparison[
        comparison["lead"].astype(str).eq("1")
        & comparison["label"].astype(str).isin(["0", "1"])
    ].sort_values(["components", "label", "ordered_minus_shuffled_mean_qlike"])
    focus.to_csv(run_dir / "lead1_focus.csv", index=False)

    metadata = {
        "chronology_run": str(args.chronology_run),
        "folds": list(map(int, args.folds)),
        "seeds": list(map(int, args.seeds)),
        "input_families": list(args.input_families),
        "reservoir_config": config,
        "pls_specs": [{"components": c, "shrinkage": s} for c, s in PLS_SPECS],
        "test_rows_used": 0,
        "notes": {
            "relative_rate": "first difference of stored log-volatility-like sequence per trading row",
            "absolute_rate": "first difference after exponentiating the stored sequence",
            "first_rate_value": "zero because the preceding observation is absent from each tensor window",
            "scaling": "channel-wise training-only standardization within each fold",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    display_columns = [
        "input_family",
        "components",
        "shrinkage",
        "lead",
        "label",
        "mean_qlike_ordered",
        "mean_qlike_shuffled",
        "ordered_minus_shuffled_mean_qlike",
        "mean_rmse_ordered",
        "mean_rmse_shuffled",
        "ordered_minus_shuffled_mean_rmse",
        "mean_qlike_p99_ordered",
        "worst_max_qlike_ordered",
    ]
    print(json.dumps(metadata, indent=2))
    print("\nLead-1 ordered-versus-shuffled focus:\n")
    print(focus[display_columns].to_string(index=False))
    print(f"\nOutputs written to {run_dir}")


if __name__ == "__main__":
    main()
