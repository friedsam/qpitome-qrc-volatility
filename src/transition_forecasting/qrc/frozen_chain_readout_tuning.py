from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.frozen_chain_readout_tools import (
    FrozenChainReadoutTuningConfig,
    _candidate_category,
    _candidate_score,
    _candidate_specs,
    _fit_correction,
    _group_metrics,
    _metadata_frame,
    _metric_payload,
    _prediction_rows,
    _select_candidate,
    chronological_inner_split,
    choose_calibration,
    gate_values,
    load_frozen_chain_archive,
    observable_indices,
)


def run_frozen_chain_readout_tuning(
    *,
    mechanism_run_dir: Path,
    results_root: Path,
    config: FrozenChainReadoutTuningConfig,
    run_id: str | None = None,
) -> Path:
    config.validate()
    mechanism_run_dir = Path(mechanism_run_dir)
    archive_path = mechanism_run_dir / "qrc_features.npz"
    archive = load_frozen_chain_archive(archive_path, case=config.case)
    indices = observable_indices(archive.feature_names, config.observable_family)
    features_all = archive.feature_matrix[:, indices]
    run_dir = begin_run(
        results_root,
        {
            "mechanism_run_dir": str(mechanism_run_dir),
            "archive_path": str(archive_path),
            "config": config.to_dict(),
            "selected_feature_names": [archive.feature_names[index] for index in indices],
        },
        run_id=run_id,
    )

    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    outer_metric_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    categories = ("shrinkage", "gating", "component", "full")

    for fold in sorted(np.unique(archive.fold)):
        fold_mask = archive.fold == int(fold)
        train = fold_mask & (archive.fold_split == "train")
        val = fold_mask & (archive.fold_split == "val")
        residual_valid = train & archive.prequential_residual_valid
        inner_fit, inner_tune = chronological_inner_split(
            archive.origin_date,
            residual_valid,
            holdout_fraction=config.inner_holdout_fraction,
        )
        y = archive.target_path
        har = archive.har_prediction_path
        residuals = archive.prequential_residual_path
        har_inner_payload = _metric_payload(y, har, inner_tune)
        specs = _candidate_specs(config, features_all.shape[1])
        fold_candidates: list[dict[str, object]] = []

        for spec in specs:
            correction = _fit_correction(
                spec,
                features=features_all,
                residuals=residuals,
                fit_mask=inner_fit,
            )
            for gate_type in config.gate_types:
                gate = gate_values(
                    gate_type,
                    encoded_sequence=archive.encoded_sequence,
                    har=har,
                    fit_mask=inner_fit,
                )
                for calibration_mode in config.calibration_modes:
                    for calibration_policy in config.selection_policies:
                        prediction, lambdas = choose_calibration(
                            correction,
                            y=y,
                            har=har,
                            tune_mask=inner_tune,
                            gate=gate,
                            mode=calibration_mode,
                            lambdas=config.global_lambdas,
                            policy=calibration_policy,
                            rmse_weight=config.balanced_rmse_weight,
                        )
                        payload = _metric_payload(y, prediction, inner_tune)
                        category_values = _candidate_category(
                            str(spec["model_kind"]),
                            str(spec["feature_spec"]),
                            gate_type,
                        )
                        row = {
                            "fold": int(fold),
                            "model_kind": spec["model_kind"],
                            "feature_spec": spec["feature_spec"],
                            "components": int(spec["components"]),
                            "alpha": spec["alpha"],
                            "gate_type": gate_type,
                            "calibration_mode": calibration_mode,
                            "calibration_policy": calibration_policy,
                            "lambda_values": json.dumps([float(value) for value in lambdas]),
                            "categories": "|".join(category_values),
                            "inner_fit_rows": int(inner_fit.sum()),
                            "inner_tune_rows": int(inner_tune.sum()),
                            **{f"inner_{key}": value for key, value in payload.items()},
                            "inner_har_qlike": float(har_inner_payload["qlike"]),
                            "inner_har_rmse": float(har_inner_payload["rmse"]),
                        }
                        for policy in config.selection_policies:
                            row[f"score_{policy}"] = _candidate_score(
                                payload,
                                har_inner_payload,
                                policy,
                                rmse_weight=config.balanced_rmse_weight,
                            )
                        fold_candidates.append(row)

        fold_frame = pd.DataFrame(fold_candidates)
        candidate_rows.extend(fold_candidates)
        metadata = _metadata_frame(archive, val)
        y_val = y[val]
        har_val = har[val]
        har_payload = _metric_payload(y, har, val)
        outer_metric_rows.append({
            "fold": int(fold),
            "model_name": "har",
            "selection_policy": "none",
            "category": "har",
            **{f"val_{key}": value for key, value in har_payload.items()},
        })
        prediction_frames.append(
            _prediction_rows(
                metadata,
                y_val,
                har_val,
                model_name="har",
                policy="none",
                category="har",
            )
        )
        group_rows.extend(
            _group_metrics(
                metadata,
                y_val,
                har_val,
                fold=int(fold),
                model_name="har",
                policy="none",
                category="har",
            )
        )

        reference_spec = {
            "model_kind": "pca",
            "feature_spec": f"prefix_{config.fixed_reference_components}",
            "indices": tuple(range(config.fixed_reference_components)),
            "components": config.fixed_reference_components,
            "alpha": config.fixed_reference_alpha,
        }
        reference_correction = _fit_correction(
            reference_spec,
            features=features_all,
            residuals=residuals,
            fit_mask=residual_valid,
        )
        reference_prediction = har + reference_correction
        reference_payload = _metric_payload(y, reference_prediction, val)
        outer_metric_rows.append({
            "fold": int(fold),
            "model_name": "frozen_reference",
            "selection_policy": "none",
            "category": "reference",
            **{f"val_{key}": value for key, value in reference_payload.items()},
        })
        prediction_frames.append(
            _prediction_rows(
                metadata,
                y_val,
                reference_prediction[val],
                model_name="frozen_reference",
                policy="none",
                category="reference",
            )
        )
        group_rows.extend(
            _group_metrics(
                metadata,
                y_val,
                reference_prediction[val],
                fold=int(fold),
                model_name="frozen_reference",
                policy="none",
                category="reference",
            )
        )

        for category in categories:
            for policy in config.selection_policies:
                chosen = _select_candidate(
                    fold_frame,
                    category=category,
                    policy=policy,
                )
                chosen_spec = next(
                    spec
                    for spec in specs
                    if spec["model_kind"] == chosen["model_kind"]
                    and spec["feature_spec"] == chosen["feature_spec"]
                    and (
                        (pd.isna(spec["alpha"]) and pd.isna(chosen["alpha"]))
                        or float(spec["alpha"]) == float(chosen["alpha"])
                    )
                )
                correction = _fit_correction(
                    chosen_spec,
                    features=features_all,
                    residuals=residuals,
                    fit_mask=residual_valid,
                )
                gate = gate_values(
                    str(chosen["gate_type"]),
                    encoded_sequence=archive.encoded_sequence,
                    har=har,
                    fit_mask=residual_valid,
                )
                lambdas = np.asarray(
                    json.loads(str(chosen["lambda_values"])),
                    dtype=float,
                )
                prediction = har + correction * gate[:, None] * lambdas[None, :]
                payload = _metric_payload(y, prediction, val)
                model_name = f"selected_{category}_{policy}"
                outer_metric_rows.append({
                    "fold": int(fold),
                    "model_name": model_name,
                    "selection_policy": policy,
                    "category": category,
                    **{f"val_{key}": value for key, value in payload.items()},
                })
                selected_rows.append({
                    "fold": int(fold),
                    "model_name": model_name,
                    "selection_policy": policy,
                    "category": category,
                    **chosen.to_dict(),
                })
                prediction_frames.append(
                    _prediction_rows(
                        metadata,
                        y_val,
                        prediction[val],
                        model_name=model_name,
                        policy=policy,
                        category=category,
                    )
                )
                group_rows.extend(
                    _group_metrics(
                        metadata,
                        y_val,
                        prediction[val],
                        fold=int(fold),
                        model_name=model_name,
                        policy=policy,
                        category=category,
                    )
                )

    candidate_frame = pd.DataFrame(candidate_rows)
    selected_frame = pd.DataFrame(selected_rows)
    outer_frame = pd.DataFrame(outer_metric_rows)
    group_frame = pd.DataFrame(group_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    candidate_frame.to_csv(run_dir / "inner_candidate_metrics.csv", index=False)
    selected_frame.to_csv(run_dir / "selected_configurations.csv", index=False)
    outer_frame.to_csv(run_dir / "outer_fold_metrics.csv", index=False)
    group_frame.to_csv(run_dir / "outer_group_metrics.csv", index=False)
    predictions.to_csv(
        run_dir / "outer_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = (
        outer_frame.groupby(
            ["model_name", "selection_policy", "category"],
            as_index=False,
        )
        .agg(
            mean_val_qlike=("val_qlike", "mean"),
            sd_val_qlike=("val_qlike", "std"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_val_mz_r2=("val_mz_r2", "mean"),
            mean_val_mz_slope=("val_mz_slope", "mean"),
            mean_val_prediction_std=("val_prediction_std", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(["mean_val_qlike", "mean_val_rmse"])
    )
    summary.to_csv(run_dir / "model_summary.csv", index=False)

    l5 = group_frame.loc[
        (group_frame["lead"] == 5) & (group_frame["label"] == 1)
    ].copy()
    l5_summary = (
        l5.groupby(
            ["model_name", "selection_policy", "category"],
            as_index=False,
        )
        .agg(
            mean_l5_qlike=("qlike", "mean"),
            mean_l5_rmse=("rmse", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values("mean_l5_qlike")
    )
    l5_summary.to_csv(run_dir / "l5_transition_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    plot = summary.sort_values("mean_val_qlike")
    ax.bar(plot["model_name"], plot["mean_val_qlike"])
    ax.set_ylabel("Mean outer validation QLIKE")
    ax.set_xlabel("Model")
    ax.set_title("Frozen-chain offline readout tuning")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(run_dir / "readout_tuning_qlike.png", dpi=180)
    plt.close(fig)

    selected_names = [
        "har",
        "frozen_reference",
        "selected_full_qlike",
        "selected_full_balanced",
    ]
    l5_predictions = predictions.loc[
        predictions["model_name"].isin(selected_names)
        & predictions["lead"].eq(5)
        & predictions["label"].eq(1)
    ]
    if not l5_predictions.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        actual = l5_predictions.groupby(
            "horizon",
            as_index=False,
        )["y_true"].mean()
        ax.plot(
            actual["horizon"],
            actual["y_true"],
            marker="o",
            label="Actual",
        )
        for model_name, local in l5_predictions.groupby("model_name"):
            mean = local.groupby("horizon", as_index=False)["y_pred"].mean()
            ax.plot(
                mean["horizon"],
                mean["y_pred"],
                marker="o",
                label=model_name,
            )
        ax.axvline(5, linestyle="--", linewidth=1)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Mean log volatility")
        ax.set_title("L5 transition forecast anatomy after readout tuning")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "l5_tuned_forecast_overlay.png", dpi=180)
        plt.close(fig)

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "development_frozen_chain_readout_tuning",
                "test_rows_used": 0,
                "mechanism_case": config.case,
                "observable_family": config.observable_family,
                "known_limitations": [
                    "The interaction strength, instability representation, and occupation family were selected on the same development folds before this assay.",
                    "Readout candidates are selected by a chronological inner holdout within each development training fold.",
                    "Outer validation folds remain untouched by this readout-selection step but are not a final unseen test set.",
                    "The gate family is deliberately limited to deterministic causal functions of input instability and the HAR forecast path.",
                ],
                "files": {
                    "inner_candidate_metrics": "inner_candidate_metrics.csv",
                    "selected_configurations": "selected_configurations.csv",
                    "outer_fold_metrics": "outer_fold_metrics.csv",
                    "outer_group_metrics": "outer_group_metrics.csv",
                    "outer_predictions": "outer_predictions.csv.gz",
                    "model_summary": "model_summary.csv",
                    "l5_transition_summary": "l5_transition_summary.csv",
                    "qlike_plot": "readout_tuning_qlike.png",
                    "l5_overlay": "l5_tuned_forecast_overlay.png",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir
