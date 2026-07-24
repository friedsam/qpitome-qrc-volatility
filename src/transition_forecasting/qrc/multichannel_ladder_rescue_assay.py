from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    probabilities_to_symmetric_modes,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.residual_head_root_cause_assay import (
    qlike_loss,
    rmse_loss,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.serial_multichannel_ladder import (
    MULTICHANNEL_CONDITIONS,
    MULTICHANNEL_REPRESENTATIONS,
    MultichannelCondition,
    MultichannelRepresentation,
    SerialMultichannelEncodingConfig,
    build_multichannel_sequences,
    evolve_serial_ladder_probe_probabilities,
    fit_three_channel_scaler,
    transform_three_channel_sequences,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


@dataclass(frozen=True)
class MultichannelLadderRescueConfig:
    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    leads: tuple[int, ...] = (1, 5, 10)
    target_lead: int = 5
    max_per_class: int = 12
    sequence_length: int = 40
    selection_seed: int = 20260722
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    path_alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    gate_cs: tuple[float, ...] = (0.1, 1.0, 10.0)
    representations: tuple[MultichannelRepresentation, ...] = MULTICHANNEL_REPRESENTATIONS
    conditions: tuple[MultichannelCondition, ...] = MULTICHANNEL_CONDITIONS
    geometry_name: str = "row_9p0um"
    incumbent_feature_indices: tuple[int, ...] = tuple(range(9))
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if self.target_lead not in self.leads:
            raise ValueError("target_lead must belong to leads")
        if self.max_per_class < 1 or self.sequence_length < 2:
            raise ValueError("sample cap and sequence length must be positive")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not self.path_alphas or any(value <= 0 for value in self.path_alphas):
            raise ValueError("path_alphas must be positive")
        if not self.gate_cs or any(value <= 0 for value in self.gate_cs):
            raise ValueError("gate_cs must be positive")
        if set(self.representations).difference(MULTICHANNEL_REPRESENTATIONS):
            raise ValueError("unsupported multichannel representation")
        if set(self.conditions).difference(MULTICHANNEL_CONDITIONS):
            raise ValueError("unsupported multichannel condition")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _causal_har_design(
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
) -> np.ndarray:
    design = np.asarray(har, dtype=float).copy()
    mask = np.asarray(residual_train_mask, dtype=bool)
    design[mask] = np.asarray(y, dtype=float)[mask] - np.asarray(residuals, dtype=float)[mask]
    return design


def fit_select_direct_path(
    features: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    tune_mask: np.ndarray,
    full_fit_mask: np.ndarray,
    *,
    alphas: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    matrix = np.asarray(features, dtype=float)
    targets = np.asarray(y, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    tune = np.asarray(tune_mask, dtype=bool)
    full_fit = np.asarray(full_fit_mask, dtype=bool)
    if matrix.ndim != 2 or targets.ndim != 2 or len(matrix) != len(targets):
        raise ValueError("features and targets must be aligned matrices")
    if not fit.any() or not tune.any() or not full_fit.any():
        raise ValueError("fit, tune, and full-fit masks must be nonempty")

    rows: list[dict[str, float]] = []
    for alpha in alphas:
        scaler = StandardScaler().fit(matrix[fit])
        design = scaler.transform(matrix)
        model = Ridge(alpha=float(alpha)).fit(design[fit], targets[fit])
        prediction = np.asarray(model.predict(design), dtype=float)
        rows.append(
            {
                "alpha": float(alpha),
                "inner_qlike": qlike_loss(targets[tune], prediction[tune]),
                "inner_rmse": rmse_loss(targets[tune], prediction[tune]),
            }
        )
    selected = min(
        rows,
        key=lambda row: (row["inner_qlike"], row["inner_rmse"], row["alpha"]),
    )
    scaler = StandardScaler().fit(matrix[full_fit])
    design = scaler.transform(matrix)
    model = Ridge(alpha=float(selected["alpha"])).fit(design[full_fit], targets[full_fit])
    prediction = np.asarray(model.predict(design), dtype=float)
    return prediction, dict(selected), pd.DataFrame(rows)


def fit_select_occurrence(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    tune_mask: np.ndarray,
    full_fit_mask: np.ndarray,
    *,
    c_values: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=int)
    fit = np.asarray(fit_mask, dtype=bool)
    tune = np.asarray(tune_mask, dtype=bool)
    full_fit = np.asarray(full_fit_mask, dtype=bool)
    for name, mask in (("fit", fit), ("tune", tune), ("full_fit", full_fit)):
        if not mask.any() or set(np.unique(target[mask])) != {0, 1}:
            raise ValueError(f"{name} occurrence rows must contain both classes")

    rows: list[dict[str, float]] = []
    for c_value in c_values:
        scaler = StandardScaler().fit(matrix[fit])
        design = scaler.transform(matrix)
        model = LogisticRegression(
            C=float(c_value),
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ).fit(design[fit], target[fit])
        probability = np.asarray(model.predict_proba(design)[:, 1], dtype=float)
        rows.append(
            {
                "c_value": float(c_value),
                "inner_average_precision": float(
                    average_precision_score(target[tune], probability[tune])
                ),
                "inner_roc_auc": float(roc_auc_score(target[tune], probability[tune])),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            -row["inner_average_precision"],
            -row["inner_roc_auc"],
            row["c_value"],
        ),
    )
    scaler = StandardScaler().fit(matrix[full_fit])
    design = scaler.transform(matrix)
    model = LogisticRegression(
        C=float(selected["c_value"]),
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    ).fit(design[full_fit], target[full_fit])
    probability = np.asarray(model.predict_proba(design)[:, 1], dtype=float)
    return probability, dict(selected), pd.DataFrame(rows)


def _path_metric_rows(
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    predictions: dict[str, np.ndarray],
    validation_mask: np.ndarray,
    *,
    fold: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes = {
        "all": np.asarray(validation_mask, dtype=bool),
        "l5_control": (
            np.asarray(validation_mask, dtype=bool)
            & frame["lead"].eq(5).to_numpy()
            & frame["label"].eq(0).to_numpy()
        ),
        "l5_transition": (
            np.asarray(validation_mask, dtype=bool)
            & frame["lead"].eq(5).to_numpy()
            & frame["label"].eq(1).to_numpy()
        ),
    }
    all_predictions = {"har": np.asarray(har, dtype=float), **predictions}
    for model_name, prediction in all_predictions.items():
        for scope, mask in scopes.items():
            if not mask.any():
                continue
            rows.append(
                {
                    "fold": int(fold),
                    "model": model_name,
                    "scope": scope,
                    "samples": int(mask.sum()),
                    "qlike": qlike_loss(y[mask], prediction[mask]),
                    "rmse": rmse_loss(y[mask], prediction[mask]),
                    "mean_correction_vs_har": float(
                        np.mean(prediction[mask] - har[mask])
                    ),
                }
            )
    return rows


def _prediction_cells(
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    validation_mask: np.ndarray,
    *,
    fold: int,
) -> pd.DataFrame:
    selected = frame.loc[validation_mask].reset_index(drop=True)
    horizons = y.shape[1]
    output = pd.DataFrame(
        {
            "fold": np.repeat(int(fold), len(selected) * horizons),
            "sample_id": np.repeat(selected["sample_id"].astype(str), horizons),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "horizon": np.tile(np.arange(1, horizons + 1), len(selected)),
            "y_true": y[validation_mask].reshape(-1),
            "har_pred": har[validation_mask].reshape(-1),
        }
    )
    for model_name, prediction in predictions.items():
        output[f"pred__{model_name}"] = prediction[validation_mask].reshape(-1)
    for model_name, probability in probabilities.items():
        output[f"prob__{model_name}"] = np.repeat(
            probability[validation_mask], horizons
        )
    return output


def _pooled_occurrence_metrics(
    score_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, group in score_rows.groupby("model", sort=True):
        target = group["label"].to_numpy(dtype=int)
        score = group["probability"].to_numpy(dtype=float)
        predicted = (score >= 0.5).astype(int)
        rows.append(
            {
                "model": model_name,
                "samples": int(len(group)),
                "average_precision": float(average_precision_score(target, score)),
                "roc_auc": float(roc_auc_score(target, score)),
                "balanced_accuracy_at_0p5": float(
                    balanced_accuracy_score(target, predicted)
                ),
                "mean_control_probability": float(score[target == 0].mean()),
                "mean_transition_probability": float(score[target == 1].mean()),
            }
        )
    return pd.DataFrame(rows)


def _render_plots(
    path_metrics: pd.DataFrame,
    occurrence: pd.DataFrame,
    cells: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    pooled_path = path_metrics.groupby(["model", "scope"], as_index=False)[
        ["qlike", "rmse"]
    ].mean()
    local = pooled_path.loc[
        pooled_path["scope"].isin(["l5_control", "l5_transition"])
    ]
    pivot = local.pivot(index="model", columns="scope", values="qlike").dropna()
    if not pivot.empty:
        figure, axis = plt.subplots(figsize=(11.0, max(5.5, 0.38 * len(pivot))))
        y_position = np.arange(len(pivot))
        width = 0.38
        axis.barh(
            y_position - width / 2,
            pivot["l5_control"],
            height=width,
            label="L5 controls",
        )
        axis.barh(
            y_position + width / 2,
            pivot["l5_transition"],
            height=width,
            label="L5 transitions",
        )
        axis.set_yticks(y_position)
        axis.set_yticklabels(pivot.index)
        axis.set(title="Direct path QLIKE after input repair", xlabel="QLIKE (lower is better)")
        axis.grid(axis="x", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        filename = "l5_path_qlike.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    if not occurrence.empty:
        ordered = occurrence.sort_values("average_precision")
        figure, axis = plt.subplots(figsize=(10.0, max(5.0, 0.34 * len(ordered))))
        axis.barh(ordered["model"], ordered["average_precision"])
        axis.axvline(0.5, linestyle="--", linewidth=1.0)
        axis.set(
            title="Held-out L5 occurrence information",
            xlabel="Average precision (balanced prevalence = 0.5)",
            xlim=(0.0, 1.0),
        )
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        filename = "l5_occurrence_average_precision.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    prediction_columns = [
        column
        for column in cells.columns
        if column.startswith("pred__")
        and (
            "serial_level_rate_time_ordered__har_stack" in column
            or "serial_level_instability_slope_ordered__har_stack" in column
        )
    ]
    if prediction_columns:
        local_cells = cells.loc[cells["lead"].eq(5)].copy()
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
        for axis, label in zip(axes, (0, 1), strict=True):
            group = local_cells.loc[local_cells["label"].eq(label)]
            har_path = group.groupby("horizon")["har_pred"].mean()
            truth_path = group.groupby("horizon")["y_true"].mean()
            axis.plot(har_path.index, har_path.values, marker="o", label="HAR")
            axis.plot(truth_path.index, truth_path.values, marker="o", label="Observed")
            for column in prediction_columns:
                path = group.groupby("horizon")[column].mean()
                axis.plot(
                    path.index,
                    path.values,
                    marker="o",
                    label=column.replace("pred__", "").replace("__har_stack", ""),
                )
            axis.axvline(5, linestyle="--", linewidth=1.0)
            axis.set_title("L5 controls" if label == 0 else "L5 transitions")
            axis.set_xlabel("Forecast horizon")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Log-volatility path")
        axes[1].legend(fontsize=7)
        figure.tight_layout()
        filename = "l5_forecast_paths.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    control_models = [
        model
        for model in path_metrics["model"].unique()
        if model.startswith("serial_") and model.endswith("__har_stack")
    ]
    rows: list[dict[str, object]] = []
    for model in control_models:
        if "_ordered__" not in model:
            continue
        base = model.replace("_ordered__", "_{condition}__")
        ordered = path_metrics.loc[
            path_metrics["model"].eq(model)
            & path_metrics["scope"].eq("l5_transition")
        ]["qlike"].mean()
        for condition in ("shuffled", "interaction_off"):
            control_name = base.format(condition=condition)
            control = path_metrics.loc[
                path_metrics["model"].eq(control_name)
                & path_metrics["scope"].eq("l5_transition")
            ]["qlike"]
            if not control.empty:
                rows.append(
                    {
                        "model": model.replace("_ordered__har_stack", ""),
                        "condition": condition,
                        "control_minus_ordered_qlike": float(control.mean() - ordered),
                    }
                )
    controls = pd.DataFrame(rows)
    if not controls.empty:
        pivot = controls.pivot(
            index="model", columns="condition", values="control_minus_ordered_qlike"
        ).fillna(0.0)
        figure, axis = plt.subplots(figsize=(9.0, 5.2))
        pivot.plot(kind="bar", ax=axis)
        axis.axhline(0.0, linewidth=1.0)
        axis.set(
            title="Does ordered interacting evolution help L5 transitions?",
            ylabel="Control QLIKE minus ordered QLIKE (positive favors ordered)",
            xlabel="Encoding",
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        filename = "ordered_control_qlike_deltas.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)
    return outputs


def run_multichannel_ladder_rescue_assay(
    *,
    fold_dir: Path,
    source_spacing_run: Path,
    results_root: Path,
    config: MultichannelLadderRescueConfig,
    encoding: SerialMultichannelEncodingConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float = 1.25,
    run_id: str | None = None,
) -> Path:
    config.validate()
    encoding.validate()
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("rescue assay requires an exact six-atom ladder")
    cache_root = Path(source_spacing_run) / "probability_cache" / config.geometry_name
    if not cache_root.is_dir():
        raise FileNotFoundError(f"missing incumbent probability cache: {cache_root}")

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "source_spacing_run": str(source_spacing_run),
            "config": config.to_dict(),
            "encoding": encoding.to_dict(),
            "reservoir": reservoir.to_dict(),
            "geometry": geometry.to_dict(),
            "interaction_scale": float(interaction_scale),
            "scientific_question": (
                "Can physically distinct level/rate/clock or level/instability/slope "
                "encoding recover direct L5 transition and path skill lost by the "
                "two-channel HAR-residual architecture?"
            ),
            "residual_target_used": False,
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(Path(fold_dir))
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )
    path_rows: list[dict[str, object]] = []
    occurrence_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.leads,
            max_per_class=config.max_per_class,
            seed=config.selection_seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("rescue assay must not receive test rows")
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        level = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=config.sequence_length,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(level).all(axis=1)
        frame = frame.loc[usable].reset_index(drop=True)
        level = level[usable]
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        labels = frame["label"].to_numpy(dtype=int)
        l5 = frame["lead"].eq(config.target_lead).to_numpy()
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train)
        residuals, residual_train = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=config.prequential_blocks,
        )
        inner_fit, inner_tune = chronological_inner_split(
            frame["origin_date"].astype(str).to_numpy(),
            residual_train,
            holdout_fraction=config.inner_holdout_fraction,
        )
        har_design = _causal_har_design(y, har, residuals, residual_train)

        gate_fit, gate_tune = chronological_inner_split(
            frame["origin_date"].astype(str).to_numpy(),
            train & l5,
            holdout_fraction=config.inner_holdout_fraction,
        )
        gate_full_fit = train & l5
        gate_validation = validation & l5

        feature_sets: dict[str, np.ndarray] = {}
        cache_path = cache_root / f"fold_{int(fold)}.npz"
        with np.load(cache_path, allow_pickle=True) as bundle:
            incumbent_modes = np.asarray(bundle["exact_modes"], dtype=float)
            cache_ids = np.asarray(bundle["sample_id"]).astype(str)
        if not np.array_equal(cache_ids, frame["sample_id"].astype(str).to_numpy()):
            raise RuntimeError(f"fold {fold}: incumbent cache panel differs")
        feature_sets["incumbent_two_channel_ordered"] = incumbent_modes[
            :, config.incumbent_feature_indices
        ]

        raw_primary: np.ndarray | None = None
        for representation in config.representations:
            raw = build_multichannel_sequences(level, representation, encoding)
            scaler = fit_three_channel_scaler(
                raw,
                train,
                q_low=encoding.q_low,
                q_high=encoding.q_high,
            )
            scaled = transform_three_channel_sequences(raw, scaler)
            if representation == "level_rate_time":
                raw_primary = scaled.reshape(len(scaled), -1)
            for condition in config.conditions:
                probabilities, metadata = evolve_serial_ladder_probe_probabilities(
                    scaled,
                    representation,
                    reservoir,
                    geometry,
                    encoding,
                    interaction_scale=float(interaction_scale),
                    condition=condition,
                )
                feature_sets[f"serial_{representation}_{condition}"] = (
                    probabilities_to_symmetric_modes(probabilities)
                )
                cache_path_new = (
                    run_dir
                    / "probability_cache"
                    / f"serial_{representation}_{condition}"
                    / f"fold_{int(fold)}.npz"
                )
                cache_path_new.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path_new,
                    probabilities=probabilities,
                    exact_modes=feature_sets[f"serial_{representation}_{condition}"],
                    sample_id=frame["sample_id"].astype(str).to_numpy(),
                    probe_steps=np.asarray(metadata["probe_steps"], dtype=int),
                )
        if raw_primary is None:
            raise RuntimeError("level_rate_time representation is required")
        feature_sets["raw_level_rate_time"] = raw_primary

        path_predictions: dict[str, np.ndarray] = {}
        gate_probabilities: dict[str, np.ndarray] = {}
        for feature_name, features in feature_sets.items():
            for variant, design in (
                ("direct", features),
                ("har_stack", np.column_stack([har_design, features])),
            ):
                model_name = f"{feature_name}__{variant}"
                prediction, selected, candidates = fit_select_direct_path(
                    design,
                    y,
                    inner_fit,
                    inner_tune,
                    residual_train,
                    alphas=config.path_alphas,
                )
                path_predictions[model_name] = prediction
                selection_rows.append(
                    {
                        "fold": int(fold),
                        "model": model_name,
                        "task": "direct_path",
                        **selected,
                    }
                )
                candidates.to_csv(
                    run_dir
                    / "selection_candidates"
                    / f"fold_{int(fold)}__{model_name}.csv",
                    index=False,
                )

            probability, selected_gate, candidates_gate = fit_select_occurrence(
                features,
                labels,
                gate_fit,
                gate_tune,
                gate_full_fit,
                c_values=config.gate_cs,
            )
            gate_probabilities[feature_name] = probability
            selection_rows.append(
                {
                    "fold": int(fold),
                    "model": feature_name,
                    "task": "l5_occurrence",
                    **selected_gate,
                }
            )
            candidates_gate.to_csv(
                run_dir
                / "selection_candidates"
                / f"fold_{int(fold)}__{feature_name}__occurrence.csv",
                index=False,
            )
            target = labels[gate_validation]
            score = probability[gate_validation]
            occurrence_rows.append(
                {
                    "fold": int(fold),
                    "model": feature_name,
                    "samples": int(gate_validation.sum()),
                    "average_precision": float(average_precision_score(target, score)),
                    "roc_auc": float(roc_auc_score(target, score)),
                    "balanced_accuracy_at_0p5": float(
                        balanced_accuracy_score(target, (score >= 0.5).astype(int))
                    ),
                }
            )

        path_rows.extend(
            _path_metric_rows(
                frame,
                y,
                har,
                path_predictions,
                validation,
                fold=int(fold),
            )
        )
        prediction_frames.append(
            _prediction_cells(
                frame,
                y,
                har,
                path_predictions,
                gate_probabilities,
                validation,
                fold=int(fold),
            )
        )

    path_metrics = pd.DataFrame(path_rows)
    occurrence_by_fold = pd.DataFrame(occurrence_rows)
    cells = pd.concat(prediction_frames, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    score_rows: list[pd.DataFrame] = []
    sample_rows = cells.loc[cells["lead"].eq(config.target_lead)].drop_duplicates(
        ["fold", "sample_id"]
    )
    for column in [name for name in cells if name.startswith("prob__")]:
        score_rows.append(
            pd.DataFrame(
                {
                    "model": column.replace("prob__", ""),
                    "label": sample_rows["label"].to_numpy(dtype=int),
                    "probability": sample_rows[column].to_numpy(dtype=float),
                }
            )
        )
    pooled_occurrence = _pooled_occurrence_metrics(pd.concat(score_rows, ignore_index=True))
    plots = _render_plots(path_metrics, pooled_occurrence, cells, run_dir / "plots")

    path_metrics.to_csv(run_dir / "path_metrics_by_fold.csv", index=False)
    occurrence_by_fold.to_csv(run_dir / "occurrence_metrics_by_fold.csv", index=False)
    pooled_occurrence.to_csv(run_dir / "occurrence_metrics_pooled.csv", index=False)
    selections.to_csv(run_dir / "selected_hyperparameters.csv", index=False)
    cells.to_csv(run_dir / "validation_predictions.csv.gz", index=False, compression="gzip")

    pooled_path = path_metrics.groupby(["model", "scope"], as_index=False)[
        ["qlike", "rmse", "mean_correction_vs_har"]
    ].mean()
    pooled_path.to_csv(run_dir / "path_metrics_pooled.csv", index=False)
    l5_transition = pooled_path.loc[pooled_path["scope"].eq("l5_transition")]
    best_path = l5_transition.sort_values(["qlike", "rmse"]).iloc[0].to_dict()
    best_occurrence = pooled_occurrence.sort_values(
        ["average_precision", "roc_auc"], ascending=False
    ).iloc[0].to_dict()
    summary = {
        "status": "multichannel_ladder_rescue_complete",
        "test_rows_used": 0,
        "residual_target_used": False,
        "new_quantum_simulation": True,
        "best_l5_transition_path_model": best_path,
        "best_l5_occurrence_model": best_occurrence,
        "promotion_rule": (
            "Promote only a serial ordered model that improves L5 transition path metrics "
            "over HAR, separates L5 occurrence above balanced chance, and degrades under "
            "the paired shuffled or interaction-off control."
        ),
        "plots": plots,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
