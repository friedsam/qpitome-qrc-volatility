from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.l5_two_head_assay import (
    _load_cache,
    _prediction_rows,
    _reference_predictions,
    _validate_reference_parity,
    metric_table,
    qlike_loss,
    rmse_loss,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)


MODEL_LABELS = {
    "har": "HAR",
    "crisis_template_only": "Crisis template only",
    "template_plus_qrc_unit": "Template + centered QRC (lambda=1)",
    "template_plus_qrc_signed": "Template + centered QRC (signed lambda)",
}


@dataclass(frozen=True)
class L5TemplateIncrementConfig:
    folds: tuple[int, ...] = tuple(range(1, 9))
    later_folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    cache_leads: tuple[int, ...] = (1, 5, 10)
    target_lead: int = 5
    selection_seeds: tuple[int, ...] = (20260721, 20260722, 20260723)
    max_per_class: int = 12
    sequence_length: int = 40
    prequential_blocks: int = 5
    split_horizon: int = 4
    minimum_specialist_rows: int = 6
    minimum_cv_fit_rows: int = 4
    alpha_grid: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    signed_lambda_grid: tuple[float, ...] = (
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    )
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.later_folds or set(self.later_folds).difference(self.folds):
            raise ValueError("later_folds must be a nonempty subset of folds")
        if self.target_lead not in self.cache_leads:
            raise ValueError("target_lead must be present in cache_leads")
        if self.minimum_specialist_rows < 6:
            raise ValueError("minimum_specialist_rows must be at least six")
        if not 3 <= self.minimum_cv_fit_rows < self.minimum_specialist_rows:
            raise ValueError("invalid expanding-window row limits")
        if not self.alpha_grid or any(value <= 0.0 for value in self.alpha_grid):
            raise ValueError("alpha_grid must contain positive values")
        lambdas = tuple(float(value) for value in self.signed_lambda_grid)
        if 0.0 not in lambdas or not any(value < 0.0 for value in lambdas):
            raise ValueError("signed_lambda_grid must contain zero and negatives")
        if not any(value > 0.0 for value in lambdas):
            raise ValueError("signed_lambda_grid must contain positives")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fit_centered_no_intercept(
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    fit_mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(matrix, dtype=float)
    response = np.asarray(targets, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    scaler = StandardScaler().fit(features[fit])
    standardized = scaler.transform(features)
    template = response[fit].mean(axis=0)
    model = Ridge(alpha=float(alpha), fit_intercept=False)
    model.fit(standardized[fit], (response - template)[fit])
    deviation = np.asarray(model.predict(standardized), dtype=float)
    return np.asarray(template, dtype=float), deviation


def _expanding_signed_search(
    matrix: np.ndarray,
    residuals: np.ndarray,
    har: np.ndarray,
    y: np.ndarray,
    *,
    eligible: np.ndarray,
    origin_date: np.ndarray,
    horizon_slice: slice,
    config: L5TemplateIncrementConfig,
) -> tuple[float, float, pd.DataFrame]:
    dates = pd.to_datetime(
        np.asarray(origin_date).astype(str),
        format="mixed",
        utc=True,
        errors="raise",
    )
    eligible_rows = np.flatnonzero(np.asarray(eligible, dtype=bool))
    ordered = eligible_rows[np.argsort(dates[eligible_rows].asi8, kind="mergesort")]
    if len(ordered) < config.minimum_cv_fit_rows + 2:
        raise ValueError(f"only {len(ordered)} specialist rows available")

    rows: list[dict[str, float]] = []
    for alpha in config.alpha_grid:
        templates, deviations, truths, har_rows = [], [], [], []
        for stop in range(config.minimum_cv_fit_rows, len(ordered)):
            fit_mask = np.zeros(len(matrix), dtype=bool)
            fit_mask[ordered[:stop]] = True
            tune_row = int(ordered[stop])
            template, deviation = _fit_centered_no_intercept(
                matrix,
                residuals[:, horizon_slice],
                fit_mask=fit_mask,
                alpha=alpha,
            )
            templates.append(template)
            deviations.append(deviation[tune_row])
            truths.append(y[tune_row, horizon_slice])
            har_rows.append(har[tune_row, horizon_slice])
        template_matrix = np.vstack(templates)
        deviation_matrix = np.vstack(deviations)
        truth_matrix = np.vstack(truths)
        har_matrix = np.vstack(har_rows)
        for signed_lambda in config.signed_lambda_grid:
            prediction = har_matrix + template_matrix + signed_lambda * deviation_matrix
            rows.append(
                {
                    "alpha": float(alpha),
                    "signed_lambda": float(signed_lambda),
                    "inner_rmse": rmse_loss(truth_matrix, prediction),
                    "inner_qlike": qlike_loss(truth_matrix, prediction),
                    "cv_predictions": float(len(truths)),
                }
            )
    selected = min(
        rows,
        key=lambda row: (
            row["inner_rmse"],
            row["inner_qlike"],
            abs(row["signed_lambda"]),
            row["alpha"],
        ),
    )
    return selected["alpha"], selected["signed_lambda"], pd.DataFrame(rows)


def fit_template_increment_models(
    matrix: np.ndarray,
    *,
    residuals: np.ndarray,
    har: np.ndarray,
    y: np.ndarray,
    specialist_train: np.ndarray,
    origin_date: np.ndarray,
    config: L5TemplateIncrementConfig,
) -> tuple[dict[str, np.ndarray], dict[str, float], pd.DataFrame]:
    corrections = {name: [] for name in MODEL_LABELS if name != "har"}
    diagnostics: dict[str, float] = {
        "specialist_train_rows": float(np.sum(specialist_train))
    }
    candidates: list[pd.DataFrame] = []
    for head, horizon_slice in (
        ("early", slice(0, config.split_horizon)),
        ("late", slice(config.split_horizon, y.shape[1])),
    ):
        alpha, signed_lambda, search = _expanding_signed_search(
            matrix,
            residuals,
            har,
            y,
            eligible=specialist_train,
            origin_date=origin_date,
            horizon_slice=horizon_slice,
            config=config,
        )
        template, deviation = _fit_centered_no_intercept(
            matrix,
            residuals[:, horizon_slice],
            fit_mask=specialist_train,
            alpha=alpha,
        )
        repeated = np.broadcast_to(template, deviation.shape).copy()
        corrections["crisis_template_only"].append(repeated)
        corrections["template_plus_qrc_unit"].append(repeated + deviation)
        corrections["template_plus_qrc_signed"].append(
            repeated + signed_lambda * deviation
        )
        diagnostics[f"alpha_{head}"] = float(alpha)
        diagnostics[f"lambda_{head}"] = float(signed_lambda)
        diagnostics[f"template_mean_{head}"] = float(np.mean(template))
        candidates.append(search.assign(head=head))
    return (
        {name: np.column_stack(blocks) for name, blocks in corrections.items()},
        diagnostics,
        pd.concat(candidates, ignore_index=True),
    )


def _paths(predictions: pd.DataFrame, later_folds: tuple[int, ...]) -> pd.DataFrame:
    local = predictions.loc[
        predictions["fold"].isin(later_folds) & predictions["label"].eq(1)
    ].copy()
    local["correction"] = local["y_pred"] - local["har_pred"]
    local["har_residual"] = local["y_true"] - local["har_pred"]
    return (
        local.groupby(["model_name", "horizon"], sort=True)
        .agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            har_pred=("har_pred", "mean"),
            correction=("correction", "mean"),
            har_residual=("har_residual", "mean"),
        )
        .reset_index()
    )


def _plot(paths: pd.DataFrame, output_dir: Path, split_horizon: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    har = paths.loc[paths["model_name"].eq("har")].sort_values("horizon")
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.plot(
        har["horizon"],
        har["har_residual"],
        "--o",
        linewidth=2.5,
        label="Required HAR residual correction",
    )
    for model_name in MODEL_LABELS:
        if model_name == "har":
            continue
        local = paths.loc[paths["model_name"].eq(model_name)].sort_values("horizon")
        axis.plot(
            local["horizon"],
            local["correction"],
            "-o",
            label=MODEL_LABELS[model_name],
        )
    axis.axhline(0.0, linewidth=1.0)
    axis.axvline(split_horizon + 1, linestyle="--", linewidth=1.2)
    axis.set(
        title="L5 crisis template versus incremental centered QRC",
        xlabel="Forecast horizon",
        ylabel="Mean correction",
    )
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "l5_template_increment_corrections.png", dpi=220)
    plt.close(figure)


def _comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    transition = metrics.loc[metrics["group"].eq("transition")]
    template = transition.loc[
        transition["model_name"].eq("crisis_template_only")
    ][["selection_seed", "qlike", "rmse"]].rename(
        columns={"qlike": "qlike_template", "rmse": "rmse_template"}
    )
    output = []
    for model_name in ("template_plus_qrc_unit", "template_plus_qrc_signed"):
        model = transition.loc[transition["model_name"].eq(model_name)][
            ["selection_seed", "qlike", "rmse"]
        ].rename(columns={"qlike": "qlike_model", "rmse": "rmse_model"})
        merged = template.merge(model, on="selection_seed", validate="one_to_one")
        merged["model_name"] = model_name
        merged["delta_qlike_vs_template"] = (
            merged["qlike_model"] - merged["qlike_template"]
        )
        merged["delta_rmse_vs_template"] = (
            merged["rmse_model"] - merged["rmse_template"]
        )
        output.append(merged)
    return pd.concat(output, ignore_index=True)


def run_l5_template_increment_assay(
    *,
    fold_dir: Path,
    comparison_run: Path,
    results_root: Path,
    dataset_name: str = "historical",
    config: L5TemplateIncrementConfig = L5TemplateIncrementConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "comparison_run": str(comparison_run),
            "dataset_name": dataset_name,
            "assay": config.to_dict(),
            "test_rows_allowed": False,
            "new_quantum_simulation": False,
        },
        run_id=run_id,
    )
    dataset = load_rolling_fold_dataset(fold_dir)
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )
    predictions, candidates, fit_rows = [], [], []
    for seed in config.selection_seeds:
        for fold in config.folds:
            full = _select_rows_for_fold(
                dataset.manifest,
                fold=fold,
                leads=config.cache_leads,
                max_per_class=config.max_per_class,
                seed=seed,
            )
            if full["fold_split"].eq("test").any():
                raise RuntimeError("test rows are forbidden")
            tensor_rows = full["_tensor_row"].to_numpy(dtype=int)
            source = extract_level_windows(
                dataset,
                tensor_rows,
                sequence_length=config.sequence_length,
                level_channel=level_channel,
            )
            usable = dataset.valid[tensor_rows] & np.isfinite(source).all(axis=1)
            full = full.loc[usable].reset_index(drop=True)
            full_modes = _load_cache(
                comparison_run,
                dataset_name=dataset_name,
                selection_seed=seed,
                fold=fold,
                selected_ids=full["sample_id"].astype(str).to_numpy(),
            )
            y_full = full[list(TARGET_COLUMNS)].to_numpy(dtype=float)
            train_full = full["fold_split"].eq("train").to_numpy()
            har_full = _fit_har(full, y_full, train_full)
            residuals_full, residual_train_full = _prequential_har_residuals(
                full,
                y_full,
                train_full,
                blocks=config.prequential_blocks,
            )
            l5 = full["lead"].eq(config.target_lead).to_numpy()
            frame = full.loc[l5].reset_index(drop=True)
            modes, y, har = full_modes[l5], y_full[l5], har_full[l5]
            residuals, residual_train = residuals_full[l5], residual_train_full[l5]
            validation = frame["fold_split"].eq("val").to_numpy()
            specialist_train = residual_train & frame["label"].eq(1).to_numpy()
            if specialist_train.sum() < config.minimum_specialist_rows:
                raise RuntimeError(f"seed {seed} fold {fold}: too few specialist rows")
            reference = _reference_predictions(
                comparison_run,
                dataset_name=dataset_name,
                selection_seed=seed,
                fold=fold,
            )
            _validate_reference_parity(
                reference,
                frame,
                y=y,
                har=har,
                validation=validation,
            )
            predictions.append(
                _prediction_rows(
                    frame,
                    selection_seed=seed,
                    y=y,
                    prediction=har,
                    har=har,
                    validation=validation,
                    model_name="har",
                    diagnostics={},
                )
            )
            corrections, diagnostics, search = fit_template_increment_models(
                modes,
                residuals=residuals,
                har=har,
                y=y,
                specialist_train=specialist_train,
                origin_date=frame["origin_date"].astype(str).to_numpy(),
                config=config,
            )
            for model_name, correction in corrections.items():
                predictions.append(
                    _prediction_rows(
                        frame,
                        selection_seed=seed,
                        y=y,
                        prediction=har + correction,
                        har=har,
                        validation=validation,
                        model_name=model_name,
                        diagnostics=diagnostics,
                    )
                )
            candidates.append(search.assign(selection_seed=seed, fold=fold))
            fit_rows.append({"selection_seed": seed, "fold": fold, **diagnostics})

    predictions_frame = pd.concat(predictions, ignore_index=True)
    candidates_frame = pd.concat(candidates, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    metrics = metric_table(predictions_frame, config.later_folds)
    paths = _paths(predictions_frame, config.later_folds)
    comparison = _comparison(metrics)
    signed = comparison.loc[
        comparison["model_name"].eq("template_plus_qrc_signed")
    ]
    decision = {
        "status": "l5_template_increment_assay_complete",
        "test_evaluated": False,
        "new_quantum_simulation": False,
        "negative_early_lambda_fits": int((fits["lambda_early"] < 0.0).sum()),
        "negative_late_lambda_fits": int((fits["lambda_late"] < 0.0).sum()),
        "zero_early_lambda_fits": int((fits["lambda_early"] == 0.0).sum()),
        "zero_late_lambda_fits": int((fits["lambda_late"] == 0.0).sum()),
        "signed_qrc_beats_template_qlike_all_seeds": bool(
            (signed["delta_qlike_vs_template"] < 0.0).all()
        ),
        "signed_qrc_beats_template_rmse_all_seeds": bool(
            (signed["delta_rmse_vs_template"] < 0.0).all()
        ),
        "median_signed_delta_qlike_vs_template": float(
            signed["delta_qlike_vs_template"].median()
        ),
        "median_signed_delta_rmse_vs_template": float(
            signed["delta_rmse_vs_template"].median()
        ),
    }
    decision["incremental_qrc_supported"] = bool(
        decision["signed_qrc_beats_template_qlike_all_seeds"]
        and decision["signed_qrc_beats_template_rmse_all_seeds"]
    )
    predictions_frame.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    candidates_frame.to_csv(
        run_dir / "signed_alpha_lambda_candidates.csv",
        index=False,
    )
    fits.to_csv(run_dir / "selected_fit_parameters.csv", index=False)
    metrics.to_csv(run_dir / "metrics_by_seed_and_group.csv", index=False)
    comparison.to_csv(run_dir / "increment_vs_template.csv", index=False)
    paths.to_csv(run_dir / "l5_transition_paths.csv", index=False)
    (run_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    _plot(paths, run_dir / "plots", config.split_horizon)
    return run_dir
