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
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
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


MODEL_ORDER = (
    "har",
    "pooled_incumbent",
    "l5_transition_single_head",
    "l5_transition_two_head",
)
MODEL_LABELS = {
    "har": "HAR",
    "pooled_incumbent": "Pooled incumbent",
    "l5_transition_single_head": "L5-only single head",
    "l5_transition_two_head": "L5-only two head",
}


@dataclass(frozen=True)
class L5TwoHeadAssayConfig:
    folds: tuple[int, ...] = tuple(range(1, 9))
    later_folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    cache_leads: tuple[int, ...] = (1, 5, 10)
    target_lead: int = 5
    selection_seeds: tuple[int, ...] = (20260721, 20260722, 20260723)
    max_per_class: int = 12
    sequence_length: int = 40
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    split_horizon: int = 4
    alpha_grid: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    minimum_specialist_rows: int = 8
    required_sign_cells: int = 10

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if not self.later_folds or set(self.later_folds).difference(self.folds):
            raise ValueError("later_folds must be a nonempty subset of folds")
        if self.target_lead not in self.cache_leads:
            raise ValueError("target_lead must be present in cache_leads")
        if not self.selection_seeds or len(set(self.selection_seeds)) != len(
            self.selection_seeds
        ):
            raise ValueError("selection_seeds must be nonempty and unique")
        if self.max_per_class < 1 or self.minimum_specialist_rows < 4:
            raise ValueError("row limits must be positive")
        if not 1 <= self.split_horizon < 10:
            raise ValueError("split_horizon must lie in [1, 9]")
        if not self.alpha_grid or any(float(value) <= 0 for value in self.alpha_grid):
            raise ValueError("alpha_grid must contain positive values")
        total_cells = len(self.later_folds) * len(self.selection_seeds)
        if not 1 <= self.required_sign_cells <= total_cells:
            raise ValueError("required_sign_cells is outside the available cell count")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def qlike_loss(y_true: np.ndarray, prediction: np.ndarray) -> float:
    observed = np.asarray(y_true, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    difference = 2.0 * (observed - forecast)
    return float(
        np.mean(np.exp(np.clip(difference, -50.0, 50.0)) - difference - 1.0)
    )


def rmse_loss(y_true: np.ndarray, prediction: np.ndarray) -> float:
    observed = np.asarray(y_true, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean((observed - forecast) ** 2)))


def _fit_predict_ridge(
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    fit_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    features = np.asarray(matrix, dtype=float)
    response = np.asarray(targets, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if features.ndim != 2 or response.ndim != 2:
        raise ValueError("matrix and targets must both be two-dimensional")
    if len(features) != len(response) or fit.shape != (len(features),):
        raise ValueError("matrix, targets, and fit mask must align")
    if not fit.any():
        raise ValueError("fit_mask selects no rows")
    scaler = StandardScaler().fit(features[fit])
    standardized = scaler.transform(features)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(standardized[fit], response[fit])
    prediction = np.asarray(model.predict(standardized), dtype=float)
    if prediction.shape != response.shape or not np.isfinite(prediction).all():
        raise RuntimeError("ridge head returned invalid predictions")
    return prediction


def _select_alpha(
    matrix: np.ndarray,
    residuals: np.ndarray,
    har: np.ndarray,
    y: np.ndarray,
    *,
    inner_fit: np.ndarray,
    inner_tune: np.ndarray,
    horizon_slice: slice,
    alpha_grid: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for alpha in alpha_grid:
        correction = _fit_predict_ridge(
            matrix,
            residuals[:, horizon_slice],
            fit_mask=inner_fit,
            alpha=float(alpha),
        )
        prediction = har[:, horizon_slice] + correction
        tune_truth = y[inner_tune, horizon_slice]
        tune_prediction = prediction[inner_tune]
        rows.append(
            {
                "alpha": float(alpha),
                "inner_rmse": rmse_loss(tune_truth, tune_prediction),
                "inner_qlike": qlike_loss(tune_truth, tune_prediction),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            float(row["inner_rmse"]),
            float(row["inner_qlike"]),
            float(row["alpha"]),
        ),
    )
    return float(selected["alpha"]), pd.DataFrame(rows)


def fit_l5_single_head(
    matrix: np.ndarray,
    *,
    residuals: np.ndarray,
    har: np.ndarray,
    y: np.ndarray,
    specialist_train: np.ndarray,
    origin_date: np.ndarray,
    config: L5TwoHeadAssayConfig,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    inner_fit, inner_tune = chronological_inner_split(
        np.asarray(origin_date),
        np.asarray(specialist_train, dtype=bool),
        holdout_fraction=config.inner_holdout_fraction,
    )
    alpha, candidates = _select_alpha(
        matrix,
        residuals,
        har,
        y,
        inner_fit=inner_fit,
        inner_tune=inner_tune,
        horizon_slice=slice(0, y.shape[1]),
        alpha_grid=config.alpha_grid,
    )
    correction = _fit_predict_ridge(
        matrix,
        residuals,
        fit_mask=specialist_train,
        alpha=alpha,
    )
    diagnostics = {
        "alpha": alpha,
        "alpha_early": alpha,
        "alpha_late": alpha,
        "inner_fit_rows": float(inner_fit.sum()),
        "inner_tune_rows": float(inner_tune.sum()),
    }
    candidates = candidates.assign(head="single")
    return correction, diagnostics, candidates


def fit_l5_two_head(
    matrix: np.ndarray,
    *,
    residuals: np.ndarray,
    har: np.ndarray,
    y: np.ndarray,
    specialist_train: np.ndarray,
    origin_date: np.ndarray,
    config: L5TwoHeadAssayConfig,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    inner_fit, inner_tune = chronological_inner_split(
        np.asarray(origin_date),
        np.asarray(specialist_train, dtype=bool),
        holdout_fraction=config.inner_holdout_fraction,
    )
    split = int(config.split_horizon)
    early_slice = slice(0, split)
    late_slice = slice(split, y.shape[1])
    alpha_early, early_candidates = _select_alpha(
        matrix,
        residuals,
        har,
        y,
        inner_fit=inner_fit,
        inner_tune=inner_tune,
        horizon_slice=early_slice,
        alpha_grid=config.alpha_grid,
    )
    alpha_late, late_candidates = _select_alpha(
        matrix,
        residuals,
        har,
        y,
        inner_fit=inner_fit,
        inner_tune=inner_tune,
        horizon_slice=late_slice,
        alpha_grid=config.alpha_grid,
    )
    early_correction = _fit_predict_ridge(
        matrix,
        residuals[:, early_slice],
        fit_mask=specialist_train,
        alpha=alpha_early,
    )
    late_correction = _fit_predict_ridge(
        matrix,
        residuals[:, late_slice],
        fit_mask=specialist_train,
        alpha=alpha_late,
    )
    correction = np.column_stack([early_correction, late_correction])
    diagnostics = {
        "alpha": float("nan"),
        "alpha_early": alpha_early,
        "alpha_late": alpha_late,
        "inner_fit_rows": float(inner_fit.sum()),
        "inner_tune_rows": float(inner_tune.sum()),
    }
    candidates = pd.concat(
        [
            early_candidates.assign(head="early"),
            late_candidates.assign(head="late"),
        ],
        ignore_index=True,
    )
    return correction, diagnostics, candidates


def event_score_table(predictions: pd.DataFrame, split_horizon: int) -> pd.DataFrame:
    frame = predictions.copy()
    frame["correction"] = frame["y_pred"] - frame["har_pred"]
    frame["har_residual"] = frame["y_true"] - frame["har_pred"]
    keys = [
        "selection_seed",
        "fold",
        "sample_id",
        "episode_id",
        "label",
        "model_name",
    ]
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(keys, sort=True):
        early = group.loc[group["horizon"].le(split_horizon)]
        late = group.loc[group["horizon"].gt(split_horizon)]
        if early.empty or late.empty:
            continue
        early_correction = float(early["correction"].mean())
        late_correction = float(late["correction"].mean())
        early_target = float(early["har_residual"].mean())
        late_target = float(late["har_residual"].mean())
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "early_correction": early_correction,
                "late_correction": late_correction,
                "expansion": late_correction - early_correction,
                "early_target_residual": early_target,
                "late_target_residual": late_target,
                "desired_sign_pattern": bool(
                    early_correction < 0.0 and late_correction > 0.0
                ),
                "target_sign_pattern": bool(early_target < 0.0 and late_target > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    selection_seed: int,
    y: np.ndarray,
    prediction: np.ndarray,
    har: np.ndarray,
    validation: np.ndarray,
    model_name: str,
    diagnostics: dict[str, float],
) -> pd.DataFrame:
    selected = frame.loc[validation].reset_index(drop=True)
    horizons = y.shape[1]
    count = int(validation.sum()) * horizons
    return pd.DataFrame(
        {
            "selection_seed": np.repeat(int(selection_seed), count),
            "fold": np.repeat(selected["fold"].to_numpy(dtype=int), horizons),
            "sample_id": np.repeat(
                selected["sample_id"].astype(str).to_numpy(), horizons
            ),
            "episode_id": np.repeat(
                selected["episode_id"].astype(str).to_numpy(), horizons
            ),
            "origin_date": np.repeat(
                selected["origin_date"].astype(str).to_numpy(), horizons
            ),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "model_name": np.repeat(model_name, count),
            "horizon": np.tile(np.arange(1, horizons + 1), int(validation.sum())),
            "y_true": y[validation].reshape(-1),
            "y_pred": prediction[validation].reshape(-1),
            "har_pred": har[validation].reshape(-1),
            "alpha": np.repeat(float(diagnostics.get("alpha", np.nan)), count),
            "alpha_early": np.repeat(
                float(diagnostics.get("alpha_early", np.nan)), count
            ),
            "alpha_late": np.repeat(
                float(diagnostics.get("alpha_late", np.nan)), count
            ),
        }
    )


def _load_cache(
    comparison_run: Path,
    *,
    dataset_name: str,
    selection_seed: int,
    fold: int,
    selected_ids: np.ndarray,
) -> np.ndarray:
    path = (
        Path(comparison_run)
        / "probability_cache"
        / dataset_name
        / f"seed_{int(selection_seed)}"
        / f"fold_{int(fold)}.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing comparison cache: {path}")
    with np.load(path, allow_pickle=True) as bundle:
        cache_ids = np.asarray(bundle["sample_id"]).astype(str)
        modes = np.asarray(bundle["exact_modes"], dtype=float)
    if not np.array_equal(cache_ids, selected_ids):
        raise RuntimeError(f"{path}: cache IDs do not match the reconstructed panel")
    if modes.shape != (len(selected_ids), 9):
        raise RuntimeError(f"{path}: expected mode shape {(len(selected_ids), 9)}, got {modes.shape}")
    return modes


def _reference_predictions(
    comparison_run: Path,
    *,
    dataset_name: str,
    selection_seed: int,
    fold: int,
) -> pd.DataFrame:
    path = Path(comparison_run) / "predictions.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"missing comparison predictions: {path}")
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "selection_seed",
        "fold",
        "lead",
        "model_name",
        "sample_id",
        "horizon",
        "y_true",
        "y_pred",
        "har_pred",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"comparison predictions missing columns: {sorted(missing)}")
    return frame.loc[
        frame["dataset"].eq(dataset_name)
        & frame["selection_seed"].eq(int(selection_seed))
        & frame["fold"].eq(int(fold))
        & frame["lead"].eq(5)
        & frame["model_name"].isin(["har", "nine_mode_linear_segmented"])
    ].copy()


def _validate_reference_parity(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    y: np.ndarray,
    har: np.ndarray,
    validation: np.ndarray,
    atol: float = 1e-12,
) -> None:
    expected_ids = np.repeat(
        frame.loc[validation, "sample_id"].astype(str).to_numpy(), y.shape[1]
    )
    expected_horizons = np.tile(
        np.arange(1, y.shape[1] + 1), int(validation.sum())
    )
    har_rows = reference.loc[reference["model_name"].eq("har")].sort_values(
        ["sample_id", "horizon"]
    )
    expected = pd.DataFrame(
        {
            "sample_id": expected_ids,
            "horizon": expected_horizons,
            "y_true": y[validation].reshape(-1),
            "har_pred": har[validation].reshape(-1),
        }
    ).sort_values(["sample_id", "horizon"])
    if len(har_rows) != len(expected):
        raise RuntimeError("reference HAR row count differs from reconstructed validation panel")
    if not np.array_equal(
        har_rows["sample_id"].astype(str).to_numpy(),
        expected["sample_id"].astype(str).to_numpy(),
    ) or not np.array_equal(
        har_rows["horizon"].to_numpy(dtype=int),
        expected["horizon"].to_numpy(dtype=int),
    ):
        raise RuntimeError("reference HAR rows do not align with reconstructed validation rows")
    if not np.allclose(
        har_rows["y_true"].to_numpy(dtype=float),
        expected["y_true"].to_numpy(dtype=float),
        atol=atol,
        rtol=0.0,
    ):
        raise RuntimeError("reference target values differ from reconstructed targets")
    if not np.allclose(
        har_rows["har_pred"].to_numpy(dtype=float),
        expected["har_pred"].to_numpy(dtype=float),
        atol=atol,
        rtol=0.0,
    ):
        raise RuntimeError("reference HAR predictions differ from reconstructed HAR")


def _convert_reference_rows(reference: pd.DataFrame) -> pd.DataFrame:
    local = reference.copy()
    local["model_name"] = local["model_name"].replace(
        {
            "har": "har",
            "nine_mode_linear_segmented": "pooled_incumbent",
        }
    )
    for column in ("alpha", "alpha_early", "alpha_late"):
        if column not in local.columns:
            local[column] = np.nan
    return local[
        [
            "selection_seed",
            "fold",
            "sample_id",
            "episode_id",
            "origin_date",
            "label",
            "lead",
            "model_name",
            "horizon",
            "y_true",
            "y_pred",
            "har_pred",
            "alpha",
            "alpha_early",
            "alpha_late",
        ]
    ]


def metric_table(predictions: pd.DataFrame, later_folds: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    local = predictions.loc[predictions["fold"].isin(later_folds)]
    for (selection_seed, model_name, label), group in local.groupby(
        ["selection_seed", "model_name", "label"], sort=True
    ):
        rows.append(
            {
                "selection_seed": int(selection_seed),
                "model_name": str(model_name),
                "group": "transition" if int(label) == 1 else "control",
                "prediction_rows": int(len(group)),
                "episodes": int(group["episode_id"].nunique()),
                "qlike": qlike_loss(group["y_true"], group["y_pred"]),
                "rmse": rmse_loss(group["y_true"], group["y_pred"]),
                "mean_error": float((group["y_pred"] - group["y_true"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def sign_summary(
    event_scores: pd.DataFrame, later_folds: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = event_scores.loc[event_scores["fold"].isin(later_folds)].copy()
    fold_seed = (
        local.groupby(["selection_seed", "fold", "model_name", "label"], sort=True)
        .agg(
            early_correction=("early_correction", "mean"),
            late_correction=("late_correction", "mean"),
            expansion=("expansion", "mean"),
            sign_rate=("desired_sign_pattern", "mean"),
            target_sign_rate=("target_sign_pattern", "mean"),
            events=("sample_id", "nunique"),
        )
        .reset_index()
    )
    pooled = (
        local.groupby(["model_name", "label"], sort=True)
        .agg(
            early_correction=("early_correction", "mean"),
            late_correction=("late_correction", "mean"),
            expansion=("expansion", "mean"),
            sign_rate=("desired_sign_pattern", "mean"),
            target_sign_rate=("target_sign_pattern", "mean"),
            events=("sample_id", "nunique"),
        )
        .reset_index()
    )
    pooled["group"] = pooled["label"].map({0: "control", 1: "transition"})
    return fold_seed, pooled


def _path_table(predictions: pd.DataFrame, later_folds: tuple[int, ...]) -> pd.DataFrame:
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
            rows=("sample_id", "size"),
        )
        .reset_index()
    )


def _plot_paths(paths: pd.DataFrame, output_dir: Path, split_horizon: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_models = [
        "har",
        "pooled_incumbent",
        "l5_transition_single_head",
        "l5_transition_two_head",
    ]
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    har = paths.loc[paths["model_name"].eq("har")].sort_values("horizon")
    axis.plot(har["horizon"], har["y_true"], marker="o", linewidth=2.5, label="Realized")
    for model_name in selected_models:
        local = paths.loc[paths["model_name"].eq(model_name)].sort_values("horizon")
        if local.empty:
            continue
        axis.plot(
            local["horizon"],
            local["y_pred"],
            marker="o",
            linewidth=2.0,
            label=MODEL_LABELS[model_name],
        )
    axis.axvline(split_horizon + 1, linestyle="--", linewidth=1.2)
    axis.set_title("L5 transition forecast paths: pooled later folds and fixed seeds")
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Mean future log volatility")
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "l5_transition_forecast_paths.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    target = har.sort_values("horizon")
    axis.plot(
        target["horizon"],
        target["har_residual"],
        linestyle="--",
        marker="o",
        linewidth=2.5,
        label="Required HAR residual correction",
    )
    for model_name in selected_models[1:]:
        local = paths.loc[paths["model_name"].eq(model_name)].sort_values("horizon")
        if local.empty:
            continue
        axis.plot(
            local["horizon"],
            local["correction"],
            marker="o",
            linewidth=2.0,
            label=MODEL_LABELS[model_name],
        )
    axis.axhline(0.0, linewidth=1.0)
    axis.axvline(split_horizon + 1, linestyle="--", linewidth=1.2)
    axis.set_title("L5 transition correction path versus required HAR residual")
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Mean correction")
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "l5_transition_correction_paths.png", dpi=220)
    plt.close(figure)


def _plot_fold_seed_signs(
    fold_seed: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = ["l5_transition_single_head", "l5_transition_two_head"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True)
    for axis, model_name in zip(axes, models, strict=True):
        local = fold_seed.loc[
            fold_seed["model_name"].eq(model_name) & fold_seed["label"].eq(1)
        ]
        axis.scatter(local["early_correction"], local["late_correction"], s=45)
        for row in local.itertuples(index=False):
            axis.annotate(
                f"{int(row.fold)}/{int(row.selection_seed) % 100}",
                (float(row.early_correction), float(row.late_correction)),
                fontsize=7,
                alpha=0.8,
            )
        axis.axvline(0.0, linewidth=1.0)
        axis.axhline(0.0, linewidth=1.0)
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Mean early correction, h1-4")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean late correction, h5-10")
    figure.suptitle("Desired L5 resonance quadrant: early < 0 and late > 0")
    figure.tight_layout()
    figure.savefig(output_dir / "l5_transition_fold_seed_signs.png", dpi=220)
    plt.close(figure)


def run_l5_two_head_assay(
    *,
    fold_dir: Path,
    comparison_run: Path,
    results_root: Path,
    dataset_name: str = "historical",
    config: L5TwoHeadAssayConfig = L5TwoHeadAssayConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    fold_dir = Path(fold_dir)
    comparison_run = Path(comparison_run)
    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "comparison_run": str(comparison_run),
            "dataset_name": dataset_name,
            "assay": config.to_dict(),
            "scientific_test": (
                "Hold HAR and frozen nine-mode quantum features fixed; train only on "
                "L5 transition residuals and test whether independent early/late heads "
                "recover a negative h1-4 correction followed by a positive h5-10 correction."
            ),
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
    prediction_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []

    for selection_seed in config.selection_seeds:
        for fold in config.folds:
            full_frame = _select_rows_for_fold(
                dataset.manifest,
                fold=int(fold),
                leads=config.cache_leads,
                max_per_class=config.max_per_class,
                seed=int(selection_seed),
            )
            if full_frame["fold_split"].eq("test").any():
                raise RuntimeError("L5 two-head assay must not receive test rows")
            tensor_rows = full_frame["_tensor_row"].to_numpy(dtype=int)
            source = extract_level_windows(
                dataset,
                tensor_rows,
                sequence_length=config.sequence_length,
                level_channel=level_channel,
            )
            usable = dataset.valid[tensor_rows] & np.isfinite(source).all(axis=1)
            full_frame = full_frame.loc[usable].reset_index(drop=True)
            selected_ids = full_frame["sample_id"].astype(str).to_numpy()
            full_modes = _load_cache(
                comparison_run,
                dataset_name=dataset_name,
                selection_seed=int(selection_seed),
                fold=int(fold),
                selected_ids=selected_ids,
            )
            y_full = full_frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
            train_full = full_frame["fold_split"].eq("train").to_numpy()
            har_full = _fit_har(full_frame, y_full, train_full)
            residuals_full, residual_train_full = _prequential_har_residuals(
                full_frame,
                y_full,
                train_full,
                blocks=config.prequential_blocks,
            )

            l5 = full_frame["lead"].eq(config.target_lead).to_numpy()
            frame = full_frame.loc[l5].reset_index(drop=True)
            modes = full_modes[l5]
            y = y_full[l5]
            har = har_full[l5]
            residuals = residuals_full[l5]
            residual_train = residual_train_full[l5]
            validation = frame["fold_split"].eq("val").to_numpy()
            specialist_train = residual_train & frame["label"].eq(1).to_numpy()
            if int(specialist_train.sum()) < config.minimum_specialist_rows:
                raise RuntimeError(
                    f"seed {selection_seed} fold {fold}: only "
                    f"{int(specialist_train.sum())} L5 transition training rows"
                )

            reference = _reference_predictions(
                comparison_run,
                dataset_name=dataset_name,
                selection_seed=int(selection_seed),
                fold=int(fold),
            )
            _validate_reference_parity(
                reference,
                frame,
                y=y,
                har=har,
                validation=validation,
            )
            prediction_frames.append(_convert_reference_rows(reference))

            origin_dates = frame["origin_date"].astype(str).to_numpy()
            single_correction, single_diag, single_candidates = fit_l5_single_head(
                modes,
                residuals=residuals,
                har=har,
                y=y,
                specialist_train=specialist_train,
                origin_date=origin_dates,
                config=config,
            )
            two_correction, two_diag, two_candidates = fit_l5_two_head(
                modes,
                residuals=residuals,
                har=har,
                y=y,
                specialist_train=specialist_train,
                origin_date=origin_dates,
                config=config,
            )
            prediction_frames.append(
                _prediction_rows(
                    frame,
                    selection_seed=int(selection_seed),
                    y=y,
                    prediction=har + single_correction,
                    har=har,
                    validation=validation,
                    model_name="l5_transition_single_head",
                    diagnostics=single_diag,
                )
            )
            prediction_frames.append(
                _prediction_rows(
                    frame,
                    selection_seed=int(selection_seed),
                    y=y,
                    prediction=har + two_correction,
                    har=har,
                    validation=validation,
                    model_name="l5_transition_two_head",
                    diagnostics=two_diag,
                )
            )
            for model_name, candidate_frame in (
                ("l5_transition_single_head", single_candidates),
                ("l5_transition_two_head", two_candidates),
            ):
                local = candidate_frame.copy()
                local.insert(0, "selection_seed", int(selection_seed))
                local.insert(1, "fold", int(fold))
                local.insert(2, "model_name", model_name)
                candidate_frames.append(local)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.drop_duplicates(
        ["selection_seed", "fold", "sample_id", "model_name", "horizon"]
    ).reset_index(drop=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    metrics = metric_table(predictions, config.later_folds)
    event_scores = event_score_table(predictions, config.split_horizon)
    fold_seed, pooled_sign = sign_summary(event_scores, config.later_folds)
    paths = _path_table(predictions, config.later_folds)

    transition_two = pooled_sign.loc[
        pooled_sign["model_name"].eq("l5_transition_two_head")
        & pooled_sign["label"].eq(1)
    ].iloc[0]
    transition_metrics = metrics.loc[
        metrics["model_name"].eq("l5_transition_two_head")
        & metrics["group"].eq("transition")
    ]
    har_metrics = metrics.loc[
        metrics["model_name"].eq("har") & metrics["group"].eq("transition")
    ]
    metric_pair = transition_metrics.merge(
        har_metrics,
        on="selection_seed",
        suffixes=("_two", "_har"),
    )
    cell_rows = fold_seed.loc[
        fold_seed["model_name"].eq("l5_transition_two_head")
        & fold_seed["label"].eq(1)
    ]
    correct_sign_cells = int(
        ((cell_rows["early_correction"] < 0.0) & (cell_rows["late_correction"] > 0.0)).sum()
    )
    decision = {
        "status": "l5_two_head_assay_complete",
        "test_evaluated": False,
        "new_quantum_simulation": False,
        "two_head_transition_early_correction": float(
            transition_two["early_correction"]
        ),
        "two_head_transition_late_correction": float(
            transition_two["late_correction"]
        ),
        "two_head_transition_sign_rate": float(transition_two["sign_rate"]),
        "correct_sign_fold_seed_cells": correct_sign_cells,
        "available_fold_seed_cells": int(len(cell_rows)),
        "required_sign_cells": int(config.required_sign_cells),
        "qlike_improved_all_seeds": bool(
            (metric_pair["qlike_two"] < metric_pair["qlike_har"]).all()
        ),
        "rmse_improved_all_seeds": bool(
            (metric_pair["rmse_two"] < metric_pair["rmse_har"]).all()
        ),
    }
    decision["promotion_gate_passed"] = bool(
        decision["two_head_transition_early_correction"] < 0.0
        and decision["two_head_transition_late_correction"] > 0.0
        and correct_sign_cells >= config.required_sign_cells
        and decision["qlike_improved_all_seeds"]
        and decision["rmse_improved_all_seeds"]
    )

    predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    candidates.to_csv(run_dir / "inner_alpha_candidates.csv", index=False)
    metrics.to_csv(run_dir / "metrics_by_seed_and_group.csv", index=False)
    event_scores.to_csv(run_dir / "event_sign_scores.csv.gz", index=False, compression="gzip")
    fold_seed.to_csv(run_dir / "fold_seed_sign_summary.csv", index=False)
    pooled_sign.to_csv(run_dir / "pooled_sign_summary.csv", index=False)
    paths.to_csv(run_dir / "l5_transition_mean_paths.csv", index=False)
    (run_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    _plot_paths(paths, run_dir / "plots", config.split_horizon)
    _plot_fold_seed_signs(fold_seed, run_dir / "plots")
    return run_dir
