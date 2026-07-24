from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.residual_alpha_sweep_assay import fit_decomposed_ridge
from transition_forecasting.qrc.residual_head_root_cause_assay import (
    DENSITY_CURVATURE_INDICES,
    ResidualHeadRootCauseConfig,
    _select_lambdas,
    qlike_loss,
    rmse_loss,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    load_rolling_fold_dataset,
)


@dataclass(frozen=True)
class GatedL5EarlyWarningConfig:
    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    panel_leads: tuple[int, ...] = (1, 5, 10)
    target_lead: int = 5
    max_per_class: int = 12
    selection_seed: int = 20260722
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    split_horizon: int = 4
    ridge_alpha: float = 100.0
    gate_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    positive_class_weights: tuple[float, ...] = (1.0, 2.0, 4.0)
    thresholds: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    false_alert_budget: float = 0.40
    path_scales: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0, 1.25)
    early_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5)
    transition_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    geometry_name: str = "row_9p0um"

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if self.target_lead not in self.panel_leads:
            raise ValueError("target_lead must belong to panel_leads")
        if not self.gate_cs or any(float(value) <= 0 for value in self.gate_cs):
            raise ValueError("gate_cs must be positive")
        if not self.positive_class_weights or any(
            float(value) <= 0 for value in self.positive_class_weights
        ):
            raise ValueError("positive_class_weights must be positive")
        if not self.thresholds or any(not 0 < float(value) < 1 for value in self.thresholds):
            raise ValueError("thresholds must lie strictly between zero and one")
        if not 0 <= self.false_alert_budget <= 1:
            raise ValueError("false_alert_budget must lie in [0, 1]")
        if not self.path_scales or any(float(value) <= 0 for value in self.path_scales):
            raise ValueError("path_scales must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def lambda_config(self) -> ResidualHeadRootCauseConfig:
        return ResidualHeadRootCauseConfig(
            folds=self.folds,
            later_folds=self.folds,
            leads=self.panel_leads,
            max_per_class=self.max_per_class,
            selection_seed=self.selection_seed,
            prequential_blocks=self.prequential_blocks,
            inner_holdout_fraction=self.inner_holdout_fraction,
            split_horizon=self.split_horizon,
            ridge_alpha=self.ridge_alpha,
            early_lambdas=self.early_lambdas,
            transition_lambdas=self.transition_lambdas,
            geometry_name=self.geometry_name,
        )


@dataclass(frozen=True)
class FittedGate:
    scaler: StandardScaler
    model: LogisticRegression

    def predict_probability(self, matrix: np.ndarray) -> np.ndarray:
        design = self.scaler.transform(np.asarray(matrix, dtype=float))
        return np.asarray(self.model.predict_proba(design)[:, 1], dtype=float)


def fit_gate(
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    *,
    c_value: float,
    positive_class_weight: float,
) -> FittedGate:
    features = np.asarray(matrix, dtype=float)
    target = np.asarray(labels, dtype=int)
    fit = np.asarray(fit_mask, dtype=bool)
    if features.ndim != 2 or target.shape != (len(features),):
        raise ValueError("gate feature matrix and labels must align")
    if fit.shape != (len(features),) or not fit.any():
        raise ValueError("fit_mask must select aligned rows")
    if set(np.unique(target[fit])) != {0, 1}:
        raise ValueError("gate fitting rows must contain both classes")
    scaler = StandardScaler().fit(features[fit])
    design = scaler.transform(features)
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        class_weight={0: 1.0, 1: float(positive_class_weight)},
        max_iter=2000,
        random_state=0,
    ).fit(design[fit], target[fit])
    return FittedGate(scaler=scaler, model=model)


def gate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    target = np.asarray(labels, dtype=int)
    score = np.asarray(probabilities, dtype=float)
    predicted = (score >= float(threshold)).astype(int)
    negative = target == 0
    return {
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "false_alert_rate": float((predicted[negative] == 1).mean()) if negative.any() else float("nan"),
        "average_precision": float(average_precision_score(target, score)),
        "roc_auc": float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else float("nan"),
    }


def select_gate_configuration(
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    tune_mask: np.ndarray,
    config: GatedL5EarlyWarningConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for c_value in config.gate_cs:
        for positive_weight in config.positive_class_weights:
            fitted = fit_gate(
                matrix,
                labels,
                fit_mask,
                c_value=float(c_value),
                positive_class_weight=float(positive_weight),
            )
            probabilities = fitted.predict_probability(matrix[tune_mask])
            target = labels[tune_mask]
            for threshold in config.thresholds:
                rows.append(
                    {
                        "c_value": float(c_value),
                        "positive_class_weight": float(positive_weight),
                        "threshold": float(threshold),
                        **gate_metrics(target, probabilities, float(threshold)),
                    }
                )
    frame = pd.DataFrame(rows)
    feasible = frame.loc[frame["false_alert_rate"] <= config.false_alert_budget]
    pool = feasible if not feasible.empty else frame
    selected = min(
        (row for _, row in pool.iterrows()),
        key=lambda row: (
            -float(row["recall"]),
            -float(row["precision"]),
            -float(row["average_precision"]),
            float(row["false_alert_rate"]),
            float(row["c_value"]),
            float(row["positive_class_weight"]),
            float(row["threshold"]),
        ),
    )
    return {key: float(value) for key, value in selected.items()}, frame


def conditional_paths(
    residuals: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    current_total: np.ndarray,
    split_horizon: int,
) -> dict[str, np.ndarray]:
    positive = np.asarray(fit_mask, dtype=bool) & (np.asarray(labels, dtype=int) == 1)
    if not positive.any():
        raise ValueError("conditional path fitting requires transition rows")
    mean_path = np.asarray(residuals, dtype=float)[positive].mean(axis=0)
    late_positive = np.zeros_like(mean_path)
    late_positive[int(split_horizon) :] = np.maximum(mean_path[int(split_horizon) :], 0.0)
    return {
        "transition_mean": mean_path,
        "late_positive_mean": late_positive,
        "incumbent_total": np.asarray(current_total, dtype=float),
    }


def correction_from_candidate(
    probabilities: np.ndarray,
    base_path: np.ndarray,
    *,
    threshold: float,
    activation: str,
    scale: float,
) -> np.ndarray:
    score = np.asarray(probabilities, dtype=float)
    path = np.asarray(base_path, dtype=float)
    if activation == "hard":
        gate = (score >= float(threshold)).astype(float)
    elif activation == "soft":
        gate = score
    else:
        raise ValueError(f"unsupported activation: {activation}")
    if path.ndim == 1:
        path = np.broadcast_to(path, (len(score), len(path)))
    if path.shape[0] != len(score):
        raise ValueError("base path must align with gate probabilities")
    return float(scale) * gate[:, None] * path


def select_forecast_candidate(
    y: np.ndarray,
    har: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    current_total: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    tune_mask: np.ndarray,
    gate_selection: dict[str, float],
    config: GatedL5EarlyWarningConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    paths = conditional_paths(
        residuals,
        labels,
        fit_mask,
        current_total,
        config.split_horizon,
    )
    rows: list[dict[str, object]] = []
    target = labels[tune_mask]
    truth = y[tune_mask]
    baseline = har[tune_mask]
    tune_probabilities = probabilities[tune_mask]
    for path_kind, full_path in paths.items():
        path = np.asarray(full_path, dtype=float)
        path_tune = path[tune_mask] if path.ndim == 2 else path
        for activation in ("hard", "soft"):
            for scale in config.path_scales:
                correction = correction_from_candidate(
                    tune_probabilities,
                    path_tune,
                    threshold=gate_selection["threshold"],
                    activation=activation,
                    scale=float(scale),
                )
                prediction = baseline + correction
                transition = target == 1
                control = target == 0
                negative_control = control[:, None] & ((truth - baseline) < 0.0)
                rows.append(
                    {
                        "path_kind": path_kind,
                        "activation": activation,
                        "path_scale": float(scale),
                        "transition_qlike": qlike_loss(truth[transition], prediction[transition]),
                        "transition_rmse": rmse_loss(truth[transition], prediction[transition]),
                        "control_qlike": qlike_loss(truth[control], prediction[control]),
                        "control_rmse": rmse_loss(truth[control], prediction[control]),
                        "mean_transition_correction": float(correction[transition].mean()),
                        "mean_control_correction": float(correction[control].mean()),
                        "wrong_up_control_rate": (
                            float((correction[negative_control] > 0.0).mean())
                            if negative_control.any()
                            else float("nan")
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    selected = min(
        (row for _, row in frame.iterrows()),
        key=lambda row: (
            float(row["transition_qlike"]),
            float(row["transition_rmse"]),
            float(row["control_qlike"]),
            abs(float(row["mean_control_correction"])),
            float(row["path_scale"]),
        ),
    )
    return dict(selected), frame


def _validation_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    validation_mask: np.ndarray,
    gate_probability: np.ndarray,
    gated_correction: np.ndarray,
    incumbent_correction: np.ndarray,
    *,
    fold: int,
    threshold: float,
) -> pd.DataFrame:
    selected = frame.loc[validation_mask].reset_index(drop=True)
    horizons = y.shape[1]
    truth = y[validation_mask]
    baseline = har[validation_mask]
    gated = gated_correction[validation_mask]
    incumbent = incumbent_correction[validation_mask]
    return pd.DataFrame(
        {
            "fold": np.repeat(int(fold), int(validation_mask.sum()) * horizons),
            "sample_id": np.repeat(selected["sample_id"].astype(str), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "horizon": np.tile(np.arange(1, horizons + 1), int(validation_mask.sum())),
            "gate_probability": np.repeat(gate_probability[validation_mask], horizons),
            "gate_alert": np.repeat(
                (gate_probability[validation_mask] >= float(threshold)).astype(int), horizons
            ),
            "y_true": truth.reshape(-1),
            "har_pred": baseline.reshape(-1),
            "gated_correction": gated.reshape(-1),
            "incumbent_correction": incumbent.reshape(-1),
            "gated_pred": (baseline + gated).reshape(-1),
            "incumbent_pred": (baseline + incumbent).reshape(-1),
        }
    )


def _pooled_metrics(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label_name, label in (("control", 0), ("transition", 1), ("all", None)):
        local = cells if label is None else cells.loc[cells["label"].eq(label)]
        truth = local["y_true"].to_numpy(dtype=float)
        row: dict[str, object] = {"group": label_name, "cells": int(len(local))}
        for name, column in (
            ("har", "har_pred"),
            ("incumbent", "incumbent_pred"),
            ("gated", "gated_pred"),
        ):
            prediction = local[column].to_numpy(dtype=float)
            row[f"{name}_qlike"] = qlike_loss(truth, prediction)
            row[f"{name}_rmse"] = rmse_loss(truth, prediction)
        rows.append(row)
    return pd.DataFrame(rows)


def _plots(cells: pd.DataFrame, fold_summary: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    sample_scores = cells.drop_duplicates(["fold", "sample_id"])
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    values = [
        sample_scores.loc[sample_scores["label"].eq(label), "gate_probability"].to_numpy()
        for label in (0, 1)
    ]
    axis.boxplot(values, labels=["L5 controls", "L5 transitions"], showmeans=True)
    axis.set(title="Held-out L5 gate scores", ylabel="Predicted transition probability")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    filename = "gate_score_separation.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    paths = cells.groupby(["label", "horizon"], as_index=False)[
        ["gated_correction", "incumbent_correction"]
    ].mean()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    for axis, label in zip(axes, (0, 1), strict=True):
        local = paths.loc[paths["label"].eq(label)]
        axis.plot(local["horizon"], local["incumbent_correction"], marker="o", label="incumbent")
        axis.plot(local["horizon"], local["gated_correction"], marker="o", label="gated candidate")
        axis.axhline(0.0, linewidth=1.0)
        axis.axvline(5, linestyle="--", linewidth=1.0)
        axis.set_title("L5 controls" if label == 0 else "L5 transitions")
        axis.set_xlabel("Forecast horizon")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean applied correction")
    axes[1].legend()
    figure.tight_layout()
    filename = "correction_paths.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(fold_summary))
    width = 0.35
    axis.bar(x - width / 2, fold_summary["gate_recall"], width, label="recall")
    axis.bar(x + width / 2, fold_summary["gate_false_alert_rate"], width, label="false-alert rate")
    axis.set_xticks(x)
    axis.set_xticklabels(fold_summary["fold"].astype(str))
    axis.set(
        title="Held-out gate behavior by fold",
        xlabel="Fold",
        ylabel="Rate",
        ylim=(0.0, 1.0),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "gate_recall_false_alert_by_fold.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    precision, recall, _ = precision_recall_curve(
        sample_scores["label"].to_numpy(dtype=int),
        sample_scores["gate_probability"].to_numpy(dtype=float),
    )
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.plot(recall, precision)
    axis.set(
        title="Pooled held-out L5 precision-recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    filename = "gate_precision_recall.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_gated_l5_early_warning_assay(
    *,
    fold_dir: Path,
    source_spacing_run: Path,
    results_root: Path,
    config: GatedL5EarlyWarningConfig = GatedL5EarlyWarningConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    cache_root = Path(source_spacing_run) / "probability_cache" / config.geometry_name
    if not cache_root.is_dir():
        raise FileNotFoundError(f"missing exact ladder probability cache: {cache_root}")

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "source_spacing_run": str(source_spacing_run),
            "config": config.to_dict(),
            "scientific_question": (
                "Can a learned L5 crisis gate convert the existing QRC features into a "
                "useful early-warning correction while suppressing the residual-head upward default?"
            ),
            "new_quantum_simulation": False,
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(Path(fold_dir))
    lambda_config = config.lambda_config()
    validation_frames: list[pd.DataFrame] = []
    gate_candidate_frames: list[pd.DataFrame] = []
    forecast_candidate_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.panel_leads,
            max_per_class=config.max_per_class,
            seed=config.selection_seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("gated L5 assay must not receive test rows")
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        labels = frame["label"].to_numpy(dtype=int)
        l5 = frame["lead"].eq(config.target_lead).to_numpy()
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy() & l5
        har = _fit_har(frame, y, train)
        residuals, residual_train = _prequential_har_residuals(
            frame, y, train, blocks=config.prequential_blocks
        )
        inner_fit, inner_tune = chronological_inner_split(
            frame["origin_date"].astype(str).to_numpy(),
            residual_train,
            holdout_fraction=config.inner_holdout_fraction,
        )
        gate_fit = inner_fit & l5
        gate_tune = inner_tune & l5
        full_gate_fit = residual_train & l5

        cache_path = cache_root / f"fold_{int(fold)}.npz"
        with np.load(cache_path, allow_pickle=True) as bundle:
            modes = np.asarray(bundle["exact_modes"], dtype=float)
            cache_ids = np.asarray(bundle["sample_id"]).astype(str)
        if not np.array_equal(cache_ids, frame["sample_id"].astype(str).to_numpy()):
            raise RuntimeError(f"fold {fold}: cache panel and selected panel differ")
        matrix = modes[:, DENSITY_CURVATURE_INDICES]

        gate_selection, gate_candidates = select_gate_configuration(
            matrix,
            labels,
            gate_fit,
            gate_tune,
            config,
        )
        gate_candidates.insert(0, "fold", int(fold))
        gate_candidate_frames.append(gate_candidates)
        inner_gate = fit_gate(
            matrix,
            labels,
            gate_fit,
            c_value=gate_selection["c_value"],
            positive_class_weight=gate_selection["positive_class_weight"],
        )
        inner_probabilities = inner_gate.predict_probability(matrix)

        inner_current = fit_decomposed_ridge(
            matrix, residuals, inner_fit, alpha=config.ridge_alpha
        )
        forecast_selection, forecast_candidates = select_forecast_candidate(
            y,
            har,
            labels,
            inner_probabilities,
            np.asarray(inner_current["total"], dtype=float),
            residuals,
            gate_fit,
            gate_tune,
            gate_selection,
            config,
        )
        forecast_candidates.insert(0, "fold", int(fold))
        forecast_candidate_frames.append(forecast_candidates)

        full_gate = fit_gate(
            matrix,
            labels,
            full_gate_fit,
            c_value=gate_selection["c_value"],
            positive_class_weight=gate_selection["positive_class_weight"],
        )
        full_probabilities = full_gate.predict_probability(matrix)
        full_current = fit_decomposed_ridge(
            matrix, residuals, residual_train, alpha=config.ridge_alpha
        )
        final_paths = conditional_paths(
            residuals,
            labels,
            full_gate_fit,
            np.asarray(full_current["total"], dtype=float),
            config.split_horizon,
        )
        gated_correction = correction_from_candidate(
            full_probabilities,
            final_paths[str(forecast_selection["path_kind"])],
            threshold=gate_selection["threshold"],
            activation=str(forecast_selection["activation"]),
            scale=float(forecast_selection["path_scale"]),
        )

        incumbent_early, incumbent_late, _ = _select_lambdas(
            y,
            har,
            np.asarray(inner_current["total"], dtype=float),
            inner_tune,
            lambda_config,
        )
        horizon_scale = np.full(y.shape[1], incumbent_late, dtype=float)
        horizon_scale[: config.split_horizon] = incumbent_early
        incumbent_correction = (
            np.asarray(full_current["total"], dtype=float) * horizon_scale[None, :]
        )
        validation_frames.append(
            _validation_frame(
                frame,
                y,
                har,
                validation,
                full_probabilities,
                gated_correction,
                incumbent_correction,
                fold=int(fold),
                threshold=gate_selection["threshold"],
            )
        )
        validation_gate = gate_metrics(
            labels[validation],
            full_probabilities[validation],
            gate_selection["threshold"],
        )
        fold_rows.append(
            {
                "fold": int(fold),
                **{f"gate_{key}": value for key, value in validation_gate.items()},
                **{f"selected_gate_{key}": value for key, value in gate_selection.items()},
                **{f"selected_forecast_{key}": value for key, value in forecast_selection.items()},
                "incumbent_early_lambda": float(incumbent_early),
                "incumbent_transition_lambda": float(incumbent_late),
            }
        )

    cells = pd.concat(validation_frames, ignore_index=True)
    fold_summary = pd.DataFrame(fold_rows)
    gate_candidates = pd.concat(gate_candidate_frames, ignore_index=True)
    forecast_candidates = pd.concat(forecast_candidate_frames, ignore_index=True)
    metrics = _pooled_metrics(cells)
    plots = _plots(cells, fold_summary, run_dir / "plots")

    sample_scores = cells.drop_duplicates(["fold", "sample_id"])
    pooled_gate = gate_metrics(
        sample_scores["label"].to_numpy(dtype=int),
        sample_scores["gate_probability"].to_numpy(dtype=float),
        0.5,
    )
    transition_metrics = metrics.loc[metrics["group"].eq("transition")].iloc[0]
    control_metrics = metrics.loc[metrics["group"].eq("control")].iloc[0]
    summary = {
        "status": "gated_l5_early_warning_assay_complete",
        "test_rows_used": 0,
        "new_quantum_simulation": False,
        "pooled_gate_at_0p5": pooled_gate,
        "transition_qlike": {
            "har": float(transition_metrics["har_qlike"]),
            "incumbent": float(transition_metrics["incumbent_qlike"]),
            "gated": float(transition_metrics["gated_qlike"]),
        },
        "control_qlike": {
            "har": float(control_metrics["har_qlike"]),
            "incumbent": float(control_metrics["incumbent_qlike"]),
            "gated": float(control_metrics["gated_qlike"]),
        },
        "interpretation_rule": (
            "Promotion requires held-out L5 transition recall above chance with a tolerable "
            "false-alert rate, plus a gated transition correction stronger than controls and "
            "transition QLIKE/RMSE that improve on HAR without reverting to an always-on shift."
        ),
        "plots": plots,
    }

    cells.to_csv(run_dir / "validation_cells.csv.gz", index=False, compression="gzip")
    fold_summary.to_csv(run_dir / "fold_summary.csv", index=False)
    gate_candidates.to_csv(run_dir / "gate_candidates.csv.gz", index=False, compression="gzip")
    forecast_candidates.to_csv(
        run_dir / "forecast_candidates.csv.gz", index=False, compression="gzip"
    )
    metrics.to_csv(run_dir / "pooled_metrics.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
