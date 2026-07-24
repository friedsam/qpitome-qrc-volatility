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
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
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
class ResidualAlphaSweepConfig:
    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    selection_seed: int = 20260722
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    split_horizon: int = 4
    alphas: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    early_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5)
    transition_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    geometry_name: str = "row_9p0um"

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if not self.alphas or any(float(value) <= 0 for value in self.alphas):
            raise ValueError("alphas must be positive")
        if len(set(float(value) for value in self.alphas)) != len(self.alphas):
            raise ValueError("alphas must be unique")
        if 100.0 not in self.alphas:
            raise ValueError("alphas must include the incumbent alpha=100")
        if 0.0 not in self.early_lambdas:
            raise ValueError("early_lambdas must include zero")
        if any(value < 0 for value in self.early_lambdas + self.transition_lambdas):
            raise ValueError("lambda grids must be nonnegative")
        if not 1 <= self.split_horizon < len(TARGET_COLUMNS):
            raise ValueError("split_horizon is outside the forecast path")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def lambda_config(self) -> ResidualHeadRootCauseConfig:
        return ResidualHeadRootCauseConfig(
            folds=self.folds,
            later_folds=self.folds,
            leads=self.leads,
            max_per_class=self.max_per_class,
            selection_seed=self.selection_seed,
            prequential_blocks=self.prequential_blocks,
            inner_holdout_fraction=self.inner_holdout_fraction,
            split_horizon=self.split_horizon,
            ridge_alpha=100.0,
            early_lambdas=self.early_lambdas,
            transition_lambdas=self.transition_lambdas,
            geometry_name=self.geometry_name,
        )


def fit_decomposed_ridge(
    matrix: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    """Return the exact same-fit decomposition intercept + QRC feature term."""

    features = np.asarray(matrix, dtype=float)
    targets = np.asarray(residuals, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("feature and residual matrices must be aligned")
    if fit.shape != (len(features),) or not fit.any():
        raise ValueError("fit_mask must select aligned rows")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    scaler = StandardScaler().fit(features[fit])
    design = scaler.transform(features)
    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(design[fit], targets[fit])
    coefficient = np.asarray(model.coef_, dtype=float)
    intercept_path = np.asarray(model.intercept_, dtype=float)
    feature = np.asarray(design @ coefficient.T, dtype=float)
    intercept = np.broadcast_to(intercept_path, feature.shape).copy()
    total = intercept + feature
    np.testing.assert_allclose(
        total,
        np.asarray(model.predict(design), dtype=float),
        atol=1e-11,
        rtol=1e-11,
    )
    return {
        "total": total,
        "feature": feature,
        "intercept": intercept,
        "intercept_path": intercept_path,
        "coefficient_l2": float(np.linalg.norm(coefficient)),
        "feature_rms": float(np.sqrt(np.mean(feature**2))),
        "intercept_rms": float(np.sqrt(np.mean(intercept**2))),
    }


def _flatten_validation(
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    decomposition: dict[str, np.ndarray | float],
    validation: np.ndarray,
    *,
    fold: int,
    alpha: float,
    early_lambda: float,
    late_lambda: float,
    split_horizon: int,
) -> pd.DataFrame:
    selected = frame.loc[validation].reset_index(drop=True)
    horizons = y.shape[1]
    horizon = np.tile(np.arange(1, horizons + 1), int(validation.sum()))
    scale = np.where(horizon <= split_horizon, early_lambda, late_lambda)
    truth = y[validation].reshape(-1)
    baseline = har[validation].reshape(-1)
    total = np.asarray(decomposition["total"], dtype=float)[validation].reshape(-1)
    feature = np.asarray(decomposition["feature"], dtype=float)[validation].reshape(-1)
    intercept = np.asarray(decomposition["intercept"], dtype=float)[validation].reshape(-1)
    return pd.DataFrame(
        {
            "fold": int(fold),
            "alpha": float(alpha),
            "sample_id": np.repeat(selected["sample_id"].astype(str), horizons),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "horizon": horizon,
            "segment": np.where(horizon <= split_horizon, "early", "late"),
            "segment_lambda": scale,
            "y_true": truth,
            "har_pred": baseline,
            "har_residual": truth - baseline,
            "raw_total": total,
            "raw_feature": feature,
            "raw_intercept": intercept,
            "hybrid_pred": baseline + scale * total,
            "feature_only_pred": baseline + scale * feature,
            "intercept_only_pred": baseline + scale * intercept,
        }
    )


def _metric_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = {
        "all": np.ones(len(cells), dtype=bool),
        "L5_control": (cells["lead"].eq(5) & cells["label"].eq(0)).to_numpy(),
        "L5_transition": (cells["lead"].eq(5) & cells["label"].eq(1)).to_numpy(),
    }
    segments = {
        "all": np.ones(len(cells), dtype=bool),
        "early": cells["segment"].eq("early").to_numpy(),
        "late": cells["segment"].eq("late").to_numpy(),
    }
    for alpha, _ in cells.groupby("alpha", sort=True):
        alpha_mask = cells["alpha"].eq(alpha).to_numpy()
        for group_name, group_mask in groups.items():
            for segment_name, segment_mask in segments.items():
                local = cells.loc[alpha_mask & group_mask & segment_mask]
                if local.empty:
                    continue
                truth = local["y_true"].to_numpy(dtype=float)
                row: dict[str, object] = {
                    "alpha": float(alpha),
                    "group": group_name,
                    "segment": segment_name,
                    "cells": int(len(local)),
                }
                for name, column in (
                    ("har", "har_pred"),
                    ("hybrid", "hybrid_pred"),
                    ("feature_only", "feature_only_pred"),
                    ("intercept_only", "intercept_only_pred"),
                ):
                    prediction = local[column].to_numpy(dtype=float)
                    row[f"{name}_qlike"] = qlike_loss(truth, prediction)
                    row[f"{name}_rmse"] = rmse_loss(truth, prediction)
                rows.append(row)
    return pd.DataFrame(rows)


def _direction_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["alpha", "lead", "label", "segment"]
    for values, group in cells.groupby(keys, sort=True):
        residual = group["har_residual"].to_numpy(dtype=float)
        negative = residual < 0
        total = group["raw_total"].to_numpy(dtype=float)
        feature = group["raw_feature"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "cells": int(len(group)),
                "negative_residual_cells": int(negative.sum()),
                "mean_raw_total": float(np.mean(total)),
                "mean_raw_feature": float(np.mean(feature)),
                "mean_raw_intercept": float(group["raw_intercept"].mean()),
                "wrong_up_total": (
                    float((total[negative] > 0).mean()) if negative.any() else float("nan")
                ),
                "wrong_up_feature": (
                    float((feature[negative] > 0).mean()) if negative.any() else float("nan")
                ),
                "total_residual_correlation": (
                    float(np.corrcoef(residual, total)[0, 1])
                    if np.std(residual) > 1e-12 and np.std(total) > 1e-12
                    else float("nan")
                ),
                "feature_residual_correlation": (
                    float(np.corrcoef(residual, feature)[0, 1])
                    if np.std(residual) > 1e-12 and np.std(feature) > 1e-12
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _l5_gap(direction: pd.DataFrame, column: str) -> pd.DataFrame:
    pivot = direction.loc[direction["lead"].eq(5)].pivot_table(
        index=["alpha", "segment"], columns="label", values=column, aggfunc="mean"
    ).reset_index()
    if 0 not in pivot or 1 not in pivot:
        raise RuntimeError("L5 audit requires both controls and transitions")
    pivot = pivot.rename(columns={0: "control", 1: "transition"})
    pivot["transition_minus_control"] = pivot["transition"] - pivot["control"]
    return pivot


def _plots(
    cells: pd.DataFrame,
    metrics: pd.DataFrame,
    direction: pd.DataFrame,
    fits: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    for column, title, filename in (
        (
            "mean_raw_feature",
            "L5 same-fit QRC feature gap",
            "l5_feature_gap_vs_alpha.png",
        ),
        (
            "mean_raw_total",
            "L5 total raw-correction gap",
            "l5_total_gap_vs_alpha.png",
        ),
    ):
        gap = _l5_gap(direction, column)
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        for segment, group in gap.groupby("segment", sort=True):
            axis.semilogx(
                group["alpha"], group["transition_minus_control"], marker="o", label=segment
            )
        axis.axhline(0.0, linewidth=1.0)
        axis.axvline(100.0, linestyle="--", linewidth=1.0, label="incumbent alpha")
        axis.set(
            title=title,
            xlabel="Ridge alpha",
            ylabel="Transition minus control correction",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    wrong = direction.loc[direction["lead"].eq(5) & direction["label"].eq(0)]
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for segment, group in wrong.groupby("segment", sort=True):
        axis.semilogx(group["alpha"], group["wrong_up_total"], marker="o", label=segment)
    axis.axvline(100.0, linestyle="--", linewidth=1.0, label="incumbent alpha")
    axis.set(
        title="L5 control double-punishment rate",
        xlabel="Ridge alpha",
        ylabel="P(raw correction > 0 | HAR overpredicts)",
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "l5_wrong_up_vs_alpha.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    local_metrics = metrics.loc[
        metrics["segment"].eq("all")
        & metrics["group"].isin(["L5_control", "L5_transition"])
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for group_name, group in local_metrics.groupby("group", sort=True):
        axis.semilogx(group["alpha"], group["hybrid_qlike"], marker="o", label=group_name)
    axis.axvline(100.0, linestyle="--", linewidth=1.0, label="incumbent alpha")
    axis.set(
        title="L5 hybrid QLIKE versus regularization",
        xlabel="Ridge alpha",
        ylabel="QLIKE (lower is better)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "l5_qlike_vs_alpha.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    strength = fits.groupby("alpha", as_index=False)[
        ["feature_rms", "intercept_rms"]
    ].mean()
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.semilogx(strength["alpha"], strength["feature_rms"], marker="o", label="QRC feature RMS")
    axis.semilogx(strength["alpha"], strength["intercept_rms"], marker="o", label="intercept RMS")
    axis.axvline(100.0, linestyle="--", linewidth=1.0, label="incumbent alpha")
    axis.set(
        title="Regularized QRC contribution versus unpenalized intercept",
        xlabel="Ridge alpha",
        ylabel="RMS raw correction",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "correction_strength_vs_alpha.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    shown = [alpha for alpha in (1.0, 10.0, 100.0) if alpha in set(cells["alpha"])]
    paths = cells.loc[cells["lead"].eq(5) & cells["alpha"].isin(shown)].groupby(
        ["alpha", "label", "horizon"], as_index=False
    )[["raw_total", "raw_feature"]].mean()
    for column, ylabel, filename in (
        ("raw_total", "Mean raw total correction", "l5_total_paths.png"),
        ("raw_feature", "Mean same-fit QRC feature term", "l5_feature_paths.png"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
        for axis, label in zip(axes, (0, 1), strict=True):
            for alpha, group in paths.loc[paths["label"].eq(label)].groupby("alpha"):
                axis.plot(group["horizon"], group[column], marker="o", label=f"alpha={alpha:g}")
            axis.axhline(0.0, linewidth=1.0)
            axis.axvline(5, linestyle="--", linewidth=1.0)
            axis.set_title("L5 controls" if label == 0 else "L5 transitions")
            axis.set_xlabel("Forecast horizon")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    fold_gap = cells.loc[
        cells["lead"].eq(5) & cells["segment"].eq("late")
    ].groupby(["fold", "alpha", "label"], as_index=False)["raw_feature"].mean()
    pivot = fold_gap.pivot_table(index=["fold", "alpha"], columns="label", values="raw_feature").reset_index()
    pivot["gap"] = pivot[1] - pivot[0]
    heatmap = pivot.pivot(index="fold", columns="alpha", values="gap")
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    image = axis.imshow(heatmap.to_numpy(), aspect="auto", cmap="coolwarm")
    axis.set_xticks(np.arange(len(heatmap.columns)))
    axis.set_xticklabels([f"{value:g}" for value in heatmap.columns])
    axis.set_yticks(np.arange(len(heatmap.index)))
    axis.set_yticklabels([str(value) for value in heatmap.index])
    axis.set(title="Late L5 feature gap by fold", xlabel="Ridge alpha", ylabel="Fold")
    figure.colorbar(image, ax=axis, label="Transition minus control")
    figure.tight_layout()
    filename = "l5_feature_gap_fold_heatmap.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_residual_alpha_sweep_assay(
    *,
    fold_dir: Path,
    source_spacing_run: Path,
    results_root: Path,
    config: ResidualAlphaSweepConfig = ResidualAlphaSweepConfig(),
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
                "Does alpha=100 suppress a useful L5 QRC feature response, or does "
                "weaker regularization amplify a non-discriminative/wrong-direction signal?"
            ),
            "diagnostic_only": True,
            "new_quantum_simulation": False,
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(Path(fold_dir))
    lambda_config = config.lambda_config()
    all_cells: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.leads,
            max_per_class=config.max_per_class,
            seed=config.selection_seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("alpha sweep must not receive test rows")
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        har = _fit_har(frame, y, train)
        residuals, residual_train = _prequential_har_residuals(
            frame, y, train, blocks=config.prequential_blocks
        )
        inner_fit, inner_tune = chronological_inner_split(
            frame["origin_date"].astype(str).to_numpy(),
            residual_train,
            holdout_fraction=config.inner_holdout_fraction,
        )

        cache_path = cache_root / f"fold_{int(fold)}.npz"
        with np.load(cache_path, allow_pickle=True) as bundle:
            modes = np.asarray(bundle["exact_modes"], dtype=float)
            cache_ids = np.asarray(bundle["sample_id"]).astype(str)
        if not np.array_equal(cache_ids, frame["sample_id"].astype(str).to_numpy()):
            raise RuntimeError(f"fold {fold}: cache panel and selected panel differ")
        matrix = modes[:, DENSITY_CURVATURE_INDICES]

        for alpha in config.alphas:
            inner = fit_decomposed_ridge(matrix, residuals, inner_fit, alpha=float(alpha))
            early, late, candidates = _select_lambdas(
                y,
                har,
                np.asarray(inner["total"], dtype=float),
                inner_tune,
                lambda_config,
            )
            selected = candidates.loc[
                np.isclose(candidates["early_lambda"], early)
                & np.isclose(candidates["transition_lambda"], late)
            ].iloc[0]
            candidates.insert(0, "fold", int(fold))
            candidates.insert(1, "alpha", float(alpha))
            candidate_frames.append(candidates)

            full = fit_decomposed_ridge(matrix, residuals, residual_train, alpha=float(alpha))
            all_cells.append(
                _flatten_validation(
                    frame,
                    y,
                    har,
                    full,
                    validation,
                    fold=int(fold),
                    alpha=float(alpha),
                    early_lambda=float(early),
                    late_lambda=float(late),
                    split_horizon=config.split_horizon,
                )
            )
            fit_rows.append(
                {
                    "fold": int(fold),
                    "alpha": float(alpha),
                    "selected_early_lambda": float(early),
                    "selected_transition_lambda": float(late),
                    "inner_qlike": float(selected["qlike"]),
                    "inner_rmse": float(selected["rmse"]),
                    "coefficient_l2": float(full["coefficient_l2"]),
                    "feature_rms": float(full["feature_rms"]),
                    "intercept_rms": float(full["intercept_rms"]),
                    **{
                        f"intercept_h{horizon}": float(value)
                        for horizon, value in enumerate(
                            np.asarray(full["intercept_path"], dtype=float), start=1
                        )
                    },
                }
            )

    cells = pd.concat(all_cells, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    metrics = _metric_table(cells)
    direction = _direction_table(cells)
    feature_gap = _l5_gap(direction, "mean_raw_feature")

    intercept_columns = [column for column in fits if column.startswith("intercept_h")]
    spread = fits.groupby("fold")[intercept_columns].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    max_intercept_spread = float(np.max(spread.to_numpy(dtype=float)))

    selected_rows: list[pd.Series] = []
    for _, group in fits.groupby("fold", sort=True):
        selected_rows.append(
            min(
                (row for _, row in group.iterrows()),
                key=lambda row: (
                    float(row["inner_qlike"]),
                    float(row["inner_rmse"]),
                    float(row["alpha"]),
                    float(row["selected_early_lambda"] + row["selected_transition_lambda"]),
                ),
            )
        )
    inner_selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    late_gap = feature_gap.loc[feature_gap["segment"].eq("late")]
    best_gap = late_gap.loc[late_gap["transition_minus_control"].idxmax()]
    incumbent_gap = late_gap.loc[np.isclose(late_gap["alpha"], 100.0)].iloc[0]
    plots = _plots(cells, metrics, direction, fits, run_dir / "plots")

    cells.to_csv(run_dir / "validation_cells.csv.gz", index=False, compression="gzip")
    fits.to_csv(run_dir / "fit_summary.csv", index=False)
    candidates.to_csv(run_dir / "lambda_candidates.csv.gz", index=False, compression="gzip")
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    direction.to_csv(run_dir / "direction_summary.csv", index=False)
    feature_gap.to_csv(run_dir / "l5_feature_gap.csv", index=False)
    inner_selected.to_csv(run_dir / "inner_selected_alpha_by_fold.csv", index=False)

    summary = {
        "status": "residual_alpha_sweep_complete",
        "test_rows_used": 0,
        "new_quantum_simulation": False,
        "incumbent_alpha": 100.0,
        "max_intercept_spread_across_alpha": max_intercept_spread,
        "incumbent_late_l5_feature_gap": float(incumbent_gap["transition_minus_control"]),
        "largest_late_l5_feature_gap_alpha": float(best_gap["alpha"]),
        "largest_late_l5_feature_gap": float(best_gap["transition_minus_control"]),
        "inner_selected_alphas": inner_selected[
            ["fold", "alpha", "selected_early_lambda", "selected_transition_lambda"]
        ].to_dict(orient="records"),
        "interpretation_rule": (
            "A stable positive L5 transition-minus-control feature gap at lower alpha, "
            "without unacceptable wrong-up behavior, supports over-regularization. "
            "Persistently negative gaps reject it. Mixed fold signs indicate unstable "
            "weak signal rather than a safe alpha repair."
        ),
        "plots": plots,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
