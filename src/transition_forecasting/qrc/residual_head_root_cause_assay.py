from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
from transition_forecasting.qrc.ladder_readout_upgrade_tools import (
    LadderReadoutUpgradeConfig,
    apply_calibration,
    select_inner_configuration,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    load_rolling_fold_dataset,
)

NINE_MODE_INDICES = tuple(range(9))
DENSITY_CURVATURE_INDICES = (0, 2, 3, 5, 6, 8)
MODEL_SPECS = {
    "nine_mode": NINE_MODE_INDICES,
    "density_curvature": DENSITY_CURVATURE_INDICES,
}
READOUTS = ("current_intercept", "intercept_only", "no_intercept")


@dataclass(frozen=True)
class ResidualHeadRootCauseConfig:
    folds: tuple[int, ...] = tuple(range(1, 9))
    later_folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    selection_seed: int = 20260722
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    split_horizon: int = 4
    ridge_alpha: float = 100.0
    early_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5)
    transition_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    geometry_name: str = "row_9p0um"

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if not self.later_folds or set(self.later_folds).difference(self.folds):
            raise ValueError("later_folds must be a nonempty subset of folds")
        if not self.leads or any(int(value) < 1 for value in self.leads):
            raise ValueError("leads must be positive")
        if self.max_per_class < 1 or self.prequential_blocks < 2:
            raise ValueError("sample cap and prequential block count are invalid")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not 1 <= self.split_horizon < len(TARGET_COLUMNS):
            raise ValueError("split_horizon is outside the target path")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if 0.0 not in self.early_lambdas:
            raise ValueError("early_lambdas must contain zero")
        if any(value < 0 for value in self.early_lambdas + self.transition_lambdas):
            raise ValueError("lambda grids must be nonnegative")
        if not self.geometry_name:
            raise ValueError("geometry_name cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def readout_config(self) -> LadderReadoutUpgradeConfig:
        return LadderReadoutUpgradeConfig(
            folds=self.folds,
            leads=self.leads,
            max_per_class=self.max_per_class,
            prequential_blocks=self.prequential_blocks,
            inner_holdout_fraction=self.inner_holdout_fraction,
            split_horizon=self.split_horizon,
            early_lambdas=self.early_lambdas,
            transition_lambdas=self.transition_lambdas,
            linear_alpha=self.ridge_alpha,
            seed=self.selection_seed,
        )


def qlike_loss(y_true: np.ndarray, prediction: np.ndarray) -> float:
    observed = np.asarray(y_true, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    difference = 2.0 * (observed - forecast)
    return float(np.mean(np.exp(np.clip(difference, -50.0, 50.0)) - difference - 1.0))


def rmse_loss(y_true: np.ndarray, prediction: np.ndarray) -> float:
    observed = np.asarray(y_true, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean((observed - forecast) ** 2)))


def balanced_sign_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(target, dtype=float).reshape(-1)
    estimated = np.asarray(prediction, dtype=float).reshape(-1)
    positive = truth > 0.0
    negative = truth < 0.0
    if not positive.any() or not negative.any():
        return float("nan")
    return 0.5 * (
        float((estimated[positive] > 0.0).mean())
        + float((estimated[negative] < 0.0).mean())
    )


def _fit_readouts(
    matrix: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    *,
    alpha: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features = np.asarray(matrix, dtype=float)
    targets = np.asarray(residuals, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    scaler = StandardScaler().fit(features[fit])
    standardized = scaler.transform(features)

    current = Ridge(alpha=float(alpha), fit_intercept=True).fit(
        standardized[fit], targets[fit]
    )
    current_prediction = np.asarray(current.predict(standardized), dtype=float)
    intercept = np.broadcast_to(
        np.asarray(current.intercept_, dtype=float), current_prediction.shape
    ).copy()

    no_intercept = Ridge(alpha=float(alpha), fit_intercept=False).fit(
        standardized[fit], targets[fit]
    )
    no_intercept_prediction = np.asarray(
        no_intercept.predict(standardized), dtype=float
    )
    return {
        "current_intercept": current_prediction,
        "intercept_only": intercept,
        "no_intercept": no_intercept_prediction,
    }, np.asarray(current.intercept_, dtype=float)


def _select_lambdas(
    y: np.ndarray,
    har: np.ndarray,
    correction: np.ndarray,
    tune_mask: np.ndarray,
    config: ResidualHeadRootCauseConfig,
) -> tuple[float, float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for early in config.early_lambdas:
        for late in config.transition_lambdas:
            if float(late) < float(early):
                continue
            prediction = apply_calibration(
                har,
                correction,
                early_lambda=float(early),
                transition_lambda=float(late),
                split_horizon=config.split_horizon,
            )
            rows.append(
                {
                    "early_lambda": float(early),
                    "transition_lambda": float(late),
                    "qlike": qlike_loss(y[tune_mask], prediction[tune_mask]),
                    "rmse": rmse_loss(y[tune_mask], prediction[tune_mask]),
                }
            )
    selected = min(
        rows,
        key=lambda row: (
            row["qlike"],
            row["rmse"],
            row["early_lambda"] + row["transition_lambda"],
            row["transition_lambda"],
            row["early_lambda"],
        ),
    )
    return (
        float(selected["early_lambda"]),
        float(selected["transition_lambda"]),
        pd.DataFrame(rows),
    )


def _flatten_cells(
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    correction: np.ndarray,
    row_mask: np.ndarray,
    *,
    fold: int,
    model_family: str,
    readout: str,
    population: str,
    split_horizon: int,
) -> pd.DataFrame:
    selected = frame.loc[row_mask].reset_index(drop=True)
    horizons = y.shape[1]
    truth = y[row_mask].reshape(-1)
    baseline = har[row_mask].reshape(-1)
    raw = correction[row_mask].reshape(-1)
    residual = truth - baseline
    horizon_index = np.tile(np.arange(1, horizons + 1), int(row_mask.sum()))
    return pd.DataFrame(
        {
            "fold": np.repeat(int(fold), len(truth)),
            "model_family": np.repeat(model_family, len(truth)),
            "readout": np.repeat(readout, len(truth)),
            "population": np.repeat(population, len(truth)),
            "sample_id": np.repeat(selected["sample_id"].astype(str), horizons),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "horizon": horizon_index,
            "segment": np.where(horizon_index <= split_horizon, "early", "late"),
            "y_true": truth,
            "har_pred": baseline,
            "har_residual": residual,
            "raw_correction": raw,
            "residual_sign": np.where(residual < 0.0, "negative", "positive"),
        }
    )


def _early_lambda_rows(
    cells: pd.DataFrame,
    config: ResidualHeadRootCauseConfig,
) -> pd.DataFrame:
    early = cells.loc[cells["segment"].eq("early")].copy()
    groups = {"all": np.ones(len(early), dtype=bool)}
    for lead in config.leads:
        groups[f"L{lead}"] = early["lead"].eq(int(lead)).to_numpy()
        for label in (0, 1):
            groups[f"L{lead}_label{label}"] = (
                early["lead"].eq(int(lead)) & early["label"].eq(label)
            ).to_numpy()
    for label in (0, 1):
        groups[f"label{label}"] = early["label"].eq(label).to_numpy()
    groups["residual_negative"] = early["har_residual"].lt(0.0).to_numpy()
    groups["residual_positive"] = early["har_residual"].gt(0.0).to_numpy()

    rows: list[dict[str, object]] = []
    for group_name, mask in groups.items():
        local = early.loc[mask]
        if local.empty:
            continue
        for value in config.early_lambdas:
            prediction = local["har_pred"].to_numpy(dtype=float) + float(value) * local[
                "raw_correction"
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "fold": int(local["fold"].iloc[0]),
                    "model_family": str(local["model_family"].iloc[0]),
                    "readout": str(local["readout"].iloc[0]),
                    "group": group_name,
                    "early_lambda": float(value),
                    "cells": int(len(local)),
                    "qlike": qlike_loss(local["y_true"].to_numpy(dtype=float), prediction),
                    "rmse": rmse_loss(local["y_true"].to_numpy(dtype=float), prediction),
                }
            )
    output = pd.DataFrame(rows)
    baseline = output.loc[
        output["early_lambda"].eq(0.0),
        ["fold", "model_family", "readout", "group", "qlike", "rmse"],
    ].rename(columns={"qlike": "qlike_at_zero", "rmse": "rmse_at_zero"})
    output = output.merge(
        baseline,
        on=["fold", "model_family", "readout", "group"],
        how="left",
        validate="many_to_one",
    )
    output["delta_qlike_vs_zero"] = output["qlike"] - output["qlike_at_zero"]
    output["delta_rmse_vs_zero"] = output["rmse"] - output["rmse_at_zero"]
    return output


def _direction_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["model_family", "readout", "population", "segment", "label"]
    for values, group in cells.groupby(keys, sort=True):
        residual = group["har_residual"].to_numpy(dtype=float)
        correction = group["raw_correction"].to_numpy(dtype=float)
        negative = residual < 0.0
        positive = residual > 0.0
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "cells": int(len(group)),
                "mean_har_residual": float(np.mean(residual)),
                "mean_raw_correction": float(np.mean(correction)),
                "mean_correction_when_residual_negative": (
                    float(np.mean(correction[negative])) if negative.any() else float("nan")
                ),
                "positive_correction_rate_when_residual_negative": (
                    float((correction[negative] > 0.0).mean()) if negative.any() else float("nan")
                ),
                "mean_correction_when_residual_positive": (
                    float(np.mean(correction[positive])) if positive.any() else float("nan")
                ),
                "balanced_sign_accuracy": balanced_sign_accuracy(residual, correction),
                "correlation": (
                    float(np.corrcoef(residual, correction)[0, 1])
                    if np.std(residual) > 1e-12 and np.std(correction) > 1e-12
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_plots(
    early_lambda: pd.DataFrame,
    cells: pd.DataFrame,
    intercepts: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    local = early_lambda.loc[
        early_lambda["model_family"].eq("density_curvature")
        & early_lambda["readout"].eq("current_intercept")
        & early_lambda["group"].eq("all")
    ]
    if not local.empty:
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        for fold, group in local.groupby("fold", sort=True):
            axis.plot(
                group["early_lambda"],
                group["qlike"],
                marker="o",
                label=f"fold {fold}",
            )
        axis.set(
            title="Why did inner selection choose a positive early lambda?",
            xlabel="Early lambda (h1-h4)",
            ylabel="Inner-holdout early QLIKE",
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        filename = "early_lambda_qlike_by_fold.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    grouped = (
        early_lambda.loc[
            early_lambda["model_family"].eq("density_curvature")
            & early_lambda["readout"].eq("current_intercept")
            & ~early_lambda["group"].eq("all")
        ]
        .groupby(["group", "early_lambda"], as_index=False)["delta_qlike_vs_zero"]
        .mean()
    )
    if not grouped.empty:
        pivot = grouped.pivot(
            index="group", columns="early_lambda", values="delta_qlike_vs_zero"
        )
        figure, axis = plt.subplots(figsize=(8.5, max(5.0, 0.34 * len(pivot))))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels([str(value) for value in pivot.columns])
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels(pivot.index)
        axis.set(
            title="Early-lambda QLIKE change versus lambda=0",
            xlabel="Early lambda",
            ylabel="Inner-holdout subgroup",
        )
        figure.colorbar(image, ax=axis, label="Delta QLIKE (negative is better)")
        figure.tight_layout()
        filename = "early_lambda_group_heatmap.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    path = (
        cells.loc[
            cells["model_family"].eq("density_curvature")
            & cells["population"].eq("validation")
            & cells["lead"].eq(5)
        ]
        .groupby(["readout", "label", "horizon"], as_index=False)["raw_correction"]
        .mean()
    )
    if not path.empty:
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
        for axis, label in zip(axes, (0, 1), strict=True):
            for readout, group in path.loc[path["label"].eq(label)].groupby(
                "readout", sort=True
            ):
                axis.plot(
                    group["horizon"],
                    group["raw_correction"],
                    marker="o",
                    label=readout,
                )
            axis.axhline(0.0, linewidth=1.0)
            axis.axvline(5, linestyle="--", linewidth=1.0)
            axis.set_title("L5 controls" if label == 0 else "L5 transitions")
            axis.set_xlabel("Forecast horizon")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Mean raw correction before lambda")
        axes[1].legend(fontsize=8)
        figure.suptitle("Does removing the intercept create a transition-control gap?")
        figure.tight_layout()
        filename = "l5_raw_correction_paths.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    if not intercepts.empty:
        mean_path = intercepts.groupby(
            ["model_family", "horizon"], as_index=False
        )["intercept"].mean()
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        for family, group in mean_path.groupby("model_family", sort=True):
            axis.plot(
                group["horizon"], group["intercept"], marker="o", label=family
            )
        axis.axhline(0.0, linewidth=1.0)
        axis.axvline(5, linestyle="--", linewidth=1.0)
        axis.set(
            title="Mean fitted Ridge intercept path",
            xlabel="Forecast horizon",
            ylabel="Residual-head intercept",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        filename = "ridge_intercept_path.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    return outputs


def run_residual_head_root_cause_assay(
    *,
    fold_dir: Path,
    source_spacing_run: Path,
    results_root: Path,
    config: ResidualHeadRootCauseConfig = ResidualHeadRootCauseConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    readout_config = config.readout_config()
    readout_config.validate()
    cache_root = Path(source_spacing_run) / "probability_cache" / config.geometry_name
    if not cache_root.is_dir():
        raise FileNotFoundError(f"missing exact ladder probability cache: {cache_root}")

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "source_spacing_run": str(source_spacing_run),
            "config": config.to_dict(),
            "scientific_questions": [
                "Does the inner h1-h4 panel itself prefer a positive lambda?",
                "Which lead, label, or HAR-residual-sign cells create that preference?",
                "Is the current Ridge correction dominated by its fitted intercept?",
                "Does a no-intercept readout recover an L5 transition-control response gap?",
            ],
            "new_quantum_simulation": False,
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(Path(fold_dir))
    cell_frames: list[pd.DataFrame] = []
    lambda_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    intercept_rows: list[dict[str, object]] = []

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.leads,
            max_per_class=config.max_per_class,
            seed=config.selection_seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("root-cause assay must not receive test rows")
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
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

        cache_path = cache_root / f"fold_{int(fold)}.npz"
        with np.load(cache_path, allow_pickle=True) as bundle:
            full_modes = np.asarray(bundle["exact_modes"], dtype=float)
            cache_ids = np.asarray(bundle["sample_id"]).astype(str)
        selected_ids = frame["sample_id"].astype(str).to_numpy()
        if not np.array_equal(cache_ids, selected_ids):
            raise RuntimeError(f"fold {fold}: cache panel and selected panel differ")

        for model_family, indices in MODEL_SPECS.items():
            matrix = full_modes[:, indices]
            diagnostics, _, _, _ = select_inner_configuration(
                matrix,
                y=y,
                har=har,
                residuals=residuals,
                residual_train_mask=residual_train,
                origin_date=frame["origin_date"].astype(str).to_numpy(),
                readout_kind="linear",
                calibration_kind="segmented",
                config=readout_config,
            )

            inner_predictions, inner_intercept = _fit_readouts(
                matrix,
                residuals,
                inner_fit,
                alpha=config.ridge_alpha,
            )
            full_predictions, full_intercept = _fit_readouts(
                matrix,
                residuals,
                residual_train,
                alpha=config.ridge_alpha,
            )
            for horizon, value in enumerate(full_intercept, start=1):
                intercept_rows.append(
                    {
                        "fold": int(fold),
                        "model_family": model_family,
                        "horizon": int(horizon),
                        "intercept": float(value),
                        "inner_intercept": float(inner_intercept[horizon - 1]),
                    }
                )

            for readout in READOUTS:
                early_lambda, late_lambda, candidates = _select_lambdas(
                    y,
                    har,
                    inner_predictions[readout],
                    inner_tune,
                    config,
                )
                candidates.insert(0, "fold", int(fold))
                candidates.insert(1, "model_family", model_family)
                candidates.insert(2, "readout", readout)
                lambda_frames.append(candidates)
                selection_rows.append(
                    {
                        "fold": int(fold),
                        "model_family": model_family,
                        "readout": readout,
                        "selected_early_lambda": early_lambda,
                        "selected_transition_lambda": late_lambda,
                        "production_selected_early_lambda": float(
                            diagnostics["early_lambda"]
                        ),
                        "production_selected_transition_lambda": float(
                            diagnostics["transition_lambda"]
                        ),
                    }
                )

                tune_cells = _flatten_cells(
                    frame,
                    y,
                    har,
                    inner_predictions[readout],
                    inner_tune,
                    fold=int(fold),
                    model_family=model_family,
                    readout=readout,
                    population="inner_tune",
                    split_horizon=config.split_horizon,
                )
                validation_cells = _flatten_cells(
                    frame,
                    y,
                    har,
                    full_predictions[readout],
                    validation,
                    fold=int(fold),
                    model_family=model_family,
                    readout=readout,
                    population="validation",
                    split_horizon=config.split_horizon,
                )
                cell_frames.extend([tune_cells, validation_cells])

    cells = pd.concat(cell_frames, ignore_index=True)
    lambda_candidates = pd.concat(lambda_frames, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    intercepts = pd.DataFrame(intercept_rows)
    early_lambda = pd.concat(
        [
            _early_lambda_rows(group, config)
            for _, group in cells.loc[cells["population"].eq("inner_tune")].groupby(
                ["fold", "model_family", "readout"], sort=True
            )
        ],
        ignore_index=True,
    )
    direction = _direction_summary(cells)
    plots = _render_plots(early_lambda, cells, intercepts, run_dir / "plots")

    selections.to_csv(run_dir / "selected_lambdas_by_fold.csv", index=False)
    lambda_candidates.to_csv(
        run_dir / "lambda_candidates.csv.gz", index=False, compression="gzip"
    )
    early_lambda.to_csv(run_dir / "early_lambda_subgroup_audit.csv", index=False)
    direction.to_csv(run_dir / "direction_summary.csv", index=False)
    intercepts.to_csv(run_dir / "ridge_intercepts.csv", index=False)
    cells.to_csv(
        run_dir / "raw_correction_cells.csv.gz", index=False, compression="gzip"
    )

    sparse_selection = selections.loc[
        selections["model_family"].eq("density_curvature")
        & selections["readout"].eq("current_intercept")
    ]
    sparse_direction = direction.loc[
        direction["model_family"].eq("density_curvature")
        & direction["population"].eq("validation")
        & direction["segment"].eq("early")
    ]
    current = sparse_direction.loc[
        sparse_direction["readout"].eq("current_intercept")
    ]
    no_intercept = sparse_direction.loc[
        sparse_direction["readout"].eq("no_intercept")
    ]
    summary = {
        "status": "residual_head_root_cause_assay_complete",
        "test_rows_used": 0,
        "new_quantum_simulation": False,
        "positive_production_early_lambda_folds": int(
            (sparse_selection["production_selected_early_lambda"] > 0.0).sum()
        ),
        "positive_reproduced_early_lambda_folds": int(
            (sparse_selection["selected_early_lambda"] > 0.0).sum()
        ),
        "current_early_wrong_up_rate_by_label": current[
            ["label", "positive_correction_rate_when_residual_negative"]
        ].to_dict(orient="records"),
        "no_intercept_early_wrong_up_rate_by_label": no_intercept[
            ["label", "positive_correction_rate_when_residual_negative"]
        ].to_dict(orient="records"),
        "interpretation_rule": (
            "If lambda=0 wins the early-only inner audit but production selected a positive "
            "early lambda, inspect selection/reporting parity. If L1 or residual-positive cells "
            "alone favor positive lambda, the pooled calibration population drives the choice. "
            "If no-intercept materially reduces wrong-up corrections while preserving an L5 "
            "transition-control gap, the intercept shortcut is the primary repair target."
        ),
        "plots": plots,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
