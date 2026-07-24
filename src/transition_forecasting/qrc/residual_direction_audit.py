from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run


REQUIRED_PREDICTION_COLUMNS = {
    "dataset",
    "selection_seed",
    "fold",
    "sample_id",
    "lead",
    "label",
    "model_name",
    "horizon",
    "y_true",
    "y_pred",
    "har_pred",
    "early_lambda",
    "transition_lambda",
}


@dataclass(frozen=True)
class ResidualDirectionAuditConfig:
    """Audit whether a HAR-residual model learns correction direction.

    The audit is diagnostic only. It uses saved validation predictions and never
    opens test rows or refits the quantum reservoir.
    """

    dataset: str = "historical"
    later_folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    lead: int = 5
    split_horizon: int = 4
    lambda_min: float = -2.0
    lambda_max: float = 2.0
    lambda_step: float = 0.05
    synthetic_seed: int = 20260723
    synthetic_rows: int = 600

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError("dataset cannot be empty")
        if not self.later_folds or any(int(value) < 1 for value in self.later_folds):
            raise ValueError("later_folds must be positive and nonempty")
        if self.lead < 1:
            raise ValueError("lead must be positive")
        if self.split_horizon < 1:
            raise ValueError("split_horizon must be positive")
        if not self.lambda_min < 0.0 < self.lambda_max:
            raise ValueError("lambda interval must span zero")
        if self.lambda_step <= 0.0:
            raise ValueError("lambda_step must be positive")
        if self.synthetic_rows < 100:
            raise ValueError("synthetic_rows must be at least 100")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def signed_lambda_grid(self) -> np.ndarray:
        self.validate()
        count = int(round((self.lambda_max - self.lambda_min) / self.lambda_step))
        return np.linspace(self.lambda_min, self.lambda_max, count + 1)

    def nonnegative_lambda_grid(self) -> np.ndarray:
        return self.signed_lambda_grid()[self.signed_lambda_grid() >= 0.0]


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_PREDICTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    numeric = [
        "selection_seed",
        "fold",
        "lead",
        "label",
        "horizon",
        "y_true",
        "y_pred",
        "har_pred",
    ]
    if frame.empty:
        raise ValueError("predictions are empty")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("predictions contain non-finite required values")


def augment_prediction_frame(
    frame: pd.DataFrame,
    *,
    split_horizon: int,
) -> pd.DataFrame:
    """Add residual, applied correction, direction, and harm columns."""

    validate_prediction_frame(frame)
    output = frame.copy()
    output["har_residual"] = output["y_true"] - output["har_pred"]
    output["applied_correction"] = output["y_pred"] - output["har_pred"]
    output["post_correction_residual"] = (
        output["y_true"] - output["y_pred"]
    )
    output["segment"] = np.where(
        output["horizon"].to_numpy(dtype=int) <= int(split_horizon),
        "early",
        "late",
    )
    output["segment_lambda"] = np.where(
        output["segment"].eq("early"),
        output["early_lambda"],
        output["transition_lambda"],
    )
    finite_lambda = np.isfinite(output["segment_lambda"].to_numpy(dtype=float))
    nonzero_lambda = np.abs(output["segment_lambda"].to_numpy(dtype=float)) > 1e-12
    recoverable = finite_lambda & nonzero_lambda
    raw = np.full(len(output), np.nan, dtype=float)
    raw[recoverable] = (
        output.loc[recoverable, "applied_correction"].to_numpy(dtype=float)
        / output.loc[recoverable, "segment_lambda"].to_numpy(dtype=float)
    )
    output["raw_correction_inferred"] = raw
    residual = output["har_residual"].to_numpy(dtype=float)
    correction = output["applied_correction"].to_numpy(dtype=float)
    output["direction_correct"] = np.sign(correction) == np.sign(residual)
    output["wrong_up_when_har_overpredicts"] = (residual < 0.0) & (correction > 0.0)
    output["wrong_down_when_har_underpredicts"] = (residual > 0.0) & (correction < 0.0)
    output["squared_error_change_vs_har"] = (residual - correction) ** 2 - residual**2
    output["absolute_error_change_vs_har"] = (
        np.abs(residual - correction) - np.abs(residual)
    )
    return output


def _safe_rate(mask: np.ndarray, denominator: np.ndarray) -> float:
    selected = np.asarray(denominator, dtype=bool)
    if not selected.any():
        return float("nan")
    return float(np.asarray(mask, dtype=bool)[selected].mean())


def _balanced_sign_accuracy(residual: np.ndarray, correction: np.ndarray) -> float:
    target = np.asarray(residual, dtype=float)
    predicted = np.asarray(correction, dtype=float)
    positive = target > 0.0
    negative = target < 0.0
    if not positive.any() or not negative.any():
        return float("nan")
    sensitivity = float((predicted[positive] > 0.0).mean())
    specificity = float((predicted[negative] < 0.0).mean())
    return 0.5 * (sensitivity + specificity)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) <= 1e-12 or np.std(y[finite]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def direction_metric_row(group: pd.DataFrame) -> dict[str, object]:
    residual = group["har_residual"].to_numpy(dtype=float)
    correction = group["applied_correction"].to_numpy(dtype=float)
    negative = residual < 0.0
    positive = residual > 0.0
    raw = group["raw_correction_inferred"].to_numpy(dtype=float)
    raw_finite = np.isfinite(raw)
    return {
        "rows": int(len(group)),
        "episodes": int(group["episode_id"].nunique()) if "episode_id" in group else 0,
        "negative_residual_fraction": float(negative.mean()),
        "mean_har_residual": float(np.mean(residual)),
        "mean_applied_correction": float(np.mean(correction)),
        "mean_correction_when_residual_negative": (
            float(np.mean(correction[negative])) if negative.any() else float("nan")
        ),
        "mean_correction_when_residual_positive": (
            float(np.mean(correction[positive])) if positive.any() else float("nan")
        ),
        "positive_correction_rate_when_residual_negative": _safe_rate(
            correction > 0.0, negative
        ),
        "negative_correction_rate_when_residual_positive": _safe_rate(
            correction < 0.0, positive
        ),
        "balanced_residual_sign_accuracy": _balanced_sign_accuracy(
            residual, correction
        ),
        "correction_residual_correlation": _safe_correlation(correction, residual),
        "mean_squared_error_change_vs_har": float(
            group["squared_error_change_vs_har"].mean()
        ),
        "mean_absolute_error_change_vs_har": float(
            group["absolute_error_change_vs_har"].mean()
        ),
        "raw_correction_recoverable_fraction": float(raw_finite.mean()),
        "raw_balanced_residual_sign_accuracy": (
            _balanced_sign_accuracy(residual[raw_finite], raw[raw_finite])
            if raw_finite.any()
            else float("nan")
        ),
        "raw_correction_residual_correlation": (
            _safe_correlation(raw[raw_finite], residual[raw_finite])
            if raw_finite.any()
            else float("nan")
        ),
        "mean_raw_correction_when_residual_negative": (
            float(np.mean(raw[raw_finite & negative]))
            if (raw_finite & negative).any()
            else float("nan")
        ),
        "mean_raw_correction_when_residual_positive": (
            float(np.mean(raw[raw_finite & positive]))
            if (raw_finite & positive).any()
            else float("nan")
        ),
    }


def direction_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "label", "segment"]
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(keys, sort=True):
        rows.append({**dict(zip(keys, values, strict=True)), **direction_metric_row(group)})
    return pd.DataFrame(rows)


def best_lambda(
    residual: np.ndarray,
    correction: np.ndarray,
    grid: Iterable[float],
) -> dict[str, float]:
    """Select a scalar by squared error on an explicitly diagnostic panel."""

    target = np.asarray(residual, dtype=float)
    signal = np.asarray(correction, dtype=float)
    finite = np.isfinite(target) & np.isfinite(signal)
    if finite.sum() < 2:
        raise ValueError("at least two finite rows are required")
    target = target[finite]
    signal = signal[finite]
    rows = []
    for value in grid:
        scaled = float(value) * signal
        rows.append(
            {
                "lambda": float(value),
                "rmse": float(np.sqrt(np.mean((target - scaled) ** 2))),
                "balanced_sign_accuracy": _balanced_sign_accuracy(target, scaled),
                "mean_squared_error_change_vs_zero": float(
                    np.mean((target - scaled) ** 2 - target**2)
                ),
            }
        )
    return min(rows, key=lambda row: (row["rmse"], abs(row["lambda"])))


def lambda_audit_table(
    frame: pd.DataFrame,
    config: ResidualDirectionAuditConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(["model_name", "segment"], sort=True):
        raw = group["raw_correction_inferred"].to_numpy(dtype=float)
        residual = group["har_residual"].to_numpy(dtype=float)
        finite = np.isfinite(raw)
        if finite.sum() < 2:
            continue
        signed = best_lambda(residual[finite], raw[finite], config.signed_lambda_grid())
        nonnegative = best_lambda(
            residual[finite], raw[finite], config.nonnegative_lambda_grid()
        )
        rows.append(
            {
                "model_name": values[0],
                "segment": values[1],
                "recoverable_rows": int(finite.sum()),
                "recoverable_fraction": float(finite.mean()),
                "best_signed_lambda": signed["lambda"],
                "best_signed_rmse": signed["rmse"],
                "best_signed_balanced_sign_accuracy": signed[
                    "balanced_sign_accuracy"
                ],
                "best_nonnegative_lambda": nonnegative["lambda"],
                "best_nonnegative_rmse": nonnegative["rmse"],
                "best_nonnegative_balanced_sign_accuracy": nonnegative[
                    "balanced_sign_accuracy"
                ],
                "signed_rmse_gain": nonnegative["rmse"] - signed["rmse"],
            }
        )
    return pd.DataFrame(rows)


def synthetic_residual_scenarios(
    *,
    rows: int,
    seed: int,
) -> pd.DataFrame:
    """Create visual sanity cases with known residual-correction relationships."""

    rng = np.random.default_rng(seed)
    residual = rng.normal(0.0, 0.45, size=rows)
    residual[np.abs(residual) < 0.05] += np.where(
        residual[np.abs(residual) < 0.05] >= 0.0, 0.08, -0.08
    )
    noise = rng.normal(0.0, 0.05, size=rows)
    scenarios = {
        "perfect_signed_signal": residual + noise,
        "globally_inverted_signal": -residual + noise,
        "upward_default_no_sign_skill": 0.25 + noise,
        "signed_signal_with_upward_bias": residual + 0.30 + noise,
    }
    frames = []
    for name, correction in scenarios.items():
        frames.append(
            pd.DataFrame(
                {
                    "scenario": name,
                    "row": np.arange(rows, dtype=int),
                    "har_residual": residual,
                    "raw_correction": correction,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def synthetic_sanity_table(
    synthetic: pd.DataFrame,
    config: ResidualDirectionAuditConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, group in synthetic.groupby("scenario", sort=True):
        residual = group["har_residual"].to_numpy(dtype=float)
        correction = group["raw_correction"].to_numpy(dtype=float)
        signed = best_lambda(residual, correction, config.signed_lambda_grid())
        nonnegative = best_lambda(
            residual, correction, config.nonnegative_lambda_grid()
        )
        rows.append(
            {
                "scenario": scenario,
                "raw_balanced_sign_accuracy": _balanced_sign_accuracy(
                    residual, correction
                ),
                "raw_correlation": _safe_correlation(residual, correction),
                "best_signed_lambda": signed["lambda"],
                "best_signed_rmse": signed["rmse"],
                "best_signed_balanced_sign_accuracy": signed[
                    "balanced_sign_accuracy"
                ],
                "best_nonnegative_lambda": nonnegative["lambda"],
                "best_nonnegative_rmse": nonnegative["rmse"],
                "best_nonnegative_balanced_sign_accuracy": nonnegative[
                    "balanced_sign_accuracy"
                ],
                "signed_rmse_gain": nonnegative["rmse"] - signed["rmse"],
            }
        )
    return pd.DataFrame(rows)


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def render_plots(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    lambda_audit: pd.DataFrame,
    synthetic: pd.DataFrame,
    synthetic_metrics: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    for scenario, group in synthetic.groupby("scenario", sort=True):
        figure, axis = plt.subplots(figsize=(6.6, 5.8))
        axis.scatter(group["har_residual"], group["raw_correction"], s=10, alpha=0.35)
        limit = float(
            max(
                np.max(np.abs(group["har_residual"])),
                np.max(np.abs(group["raw_correction"])),
            )
        )
        axis.plot([-limit, limit], [-limit, limit], linestyle="--", label="Correct sign and magnitude")
        axis.axhline(0.0, linewidth=1.0)
        axis.axvline(0.0, linewidth=1.0)
        axis.set(
            title=f"Synthetic sanity case: {scenario.replace('_', ' ')}",
            xlabel="Required HAR residual correction",
            ylabel="Model raw correction",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        filename = f"synthetic_{_safe_filename(scenario)}.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    x = np.arange(len(synthetic_metrics))
    width = 0.36
    axis.bar(
        x - width / 2,
        synthetic_metrics["best_nonnegative_lambda"],
        width,
        label="Best nonnegative lambda",
    )
    axis.bar(
        x + width / 2,
        synthetic_metrics["best_signed_lambda"],
        width,
        label="Best signed lambda",
    )
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(
        synthetic_metrics["scenario"].str.replace("_", " "), rotation=25, ha="right"
    )
    axis.set(title="What signed lambda can and cannot repair", ylabel="Selected lambda")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "synthetic_lambda_recovery.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    overall = (
        frame.groupby("model_name", sort=True)
        .apply(direction_metric_row, include_groups=False)
        .apply(pd.Series)
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(overall))
    width = 0.36
    axis.bar(
        x - width / 2,
        overall["mean_correction_when_residual_negative"],
        width,
        label="HAR residual < 0 (HAR overpredicts)",
    )
    axis.bar(
        x + width / 2,
        overall["mean_correction_when_residual_positive"],
        width,
        label="HAR residual > 0 (HAR underpredicts)",
    )
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(overall["model_name"], rotation=30, ha="right")
    axis.set(
        title="Mean applied correction conditional on the sign HAR actually needed",
        ylabel="Mean QRC correction to HAR",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "conditional_correction_by_residual_sign.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.bar(overall["model_name"], overall["balanced_residual_sign_accuracy"])
    axis.axhline(0.5, linestyle="--", linewidth=1.2, label="Chance")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticklabels(overall["model_name"], rotation=30, ha="right")
    axis.set(
        title="Can the final correction identify the sign of the HAR error?",
        ylabel="Balanced residual-sign accuracy",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "residual_sign_accuracy.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    if not lambda_audit.empty:
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        labels = lambda_audit["model_name"] + " / " + lambda_audit["segment"]
        x = np.arange(len(lambda_audit))
        width = 0.36
        axis.bar(
            x - width / 2,
            lambda_audit["best_nonnegative_lambda"],
            width,
            label="Restricted calibration",
        )
        axis.bar(
            x + width / 2,
            lambda_audit["best_signed_lambda"],
            width,
            label="Post-hoc signed diagnostic",
        )
        axis.axhline(0.0, linewidth=1.0)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set(title="Signed-lambda diagnostic on recoverable raw corrections", ylabel="Lambda")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        filename = "signed_lambda_diagnostic.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    for model_name, group in frame.groupby("model_name", sort=True):
        if len(group) > 5000:
            group = group.sample(5000, random_state=20260723)
        figure, axis = plt.subplots(figsize=(6.6, 5.8))
        axis.scatter(
            group["har_residual"],
            group["applied_correction"],
            s=10,
            alpha=0.3,
        )
        limit = float(
            max(
                np.max(np.abs(group["har_residual"])),
                np.max(np.abs(group["applied_correction"])),
            )
        )
        axis.plot([-limit, limit], [-limit, limit], linestyle="--", label="Ideal correction")
        axis.axhline(0.0, linewidth=1.0)
        axis.axvline(0.0, linewidth=1.0)
        axis.set(
            title=f"Residual direction audit: {model_name}",
            xlabel="Required correction y − HAR",
            ylabel="Applied model correction",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        filename = f"residual_vs_correction_{_safe_filename(model_name)}.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    path = (
        frame.groupby(["model_name", "label", "horizon"], sort=True, as_index=False)
        .agg(
            har_residual=("har_residual", "mean"),
            applied_correction=("applied_correction", "mean"),
        )
    )
    for model_name, group in path.groupby("model_name", sort=True):
        figure, axis = plt.subplots(figsize=(8.8, 5.8))
        for label, label_text in ((0, "Matched non-transition"), (1, "L5 transition")):
            local = group.loc[group["label"].eq(label)].sort_values("horizon")
            axis.plot(
                local["horizon"],
                local["har_residual"],
                marker="o",
                linestyle="--",
                label=f"Required: {label_text}",
            )
            axis.plot(
                local["horizon"],
                local["applied_correction"],
                marker="o",
                label=f"Applied: {label_text}",
            )
        axis.axhline(0.0, linewidth=1.0)
        axis.axvline(5, linestyle="--", linewidth=1.0)
        axis.set_xticks(range(1, 11))
        axis.set(
            title=f"Required versus applied correction path: {model_name}",
            xlabel="Forecast horizon",
            ylabel="Mean correction",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        filename = f"path_required_vs_applied_{_safe_filename(model_name)}.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    return outputs


def run_residual_direction_audit(
    *,
    predictions_path: Path,
    results_root: Path,
    config: ResidualDirectionAuditConfig = ResidualDirectionAuditConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    source = Path(predictions_path)
    if not source.is_file():
        raise FileNotFoundError(f"missing predictions file: {source}")
    predictions = pd.read_csv(source)
    validate_prediction_frame(predictions)
    local = predictions.loc[
        predictions["dataset"].eq(config.dataset)
        & predictions["fold"].isin(config.later_folds)
        & predictions["lead"].eq(config.lead)
        & ~predictions["model_name"].eq("har")
    ].copy()
    if local.empty:
        raise RuntimeError("audit selection contains no model rows")
    augmented = augment_prediction_frame(local, split_horizon=config.split_horizon)
    metrics = direction_metric_table(augmented)
    lambdas = lambda_audit_table(augmented, config)
    synthetic = synthetic_residual_scenarios(
        rows=config.synthetic_rows,
        seed=config.synthetic_seed,
    )
    synthetic_metrics = synthetic_sanity_table(synthetic, config)

    run_dir = begin_run(
        Path(results_root),
        {
            "predictions_path": str(source),
            "config": config.to_dict(),
            "test_rows_allowed": False,
            "new_quantum_simulation": False,
            "scientific_scope": (
                "Residual-direction sanity audit on saved validation predictions; "
                "not model selection and not an unbiased performance estimate."
            ),
        },
        run_id=run_id,
    )
    augmented.to_csv(run_dir / "augmented_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(run_dir / "direction_metrics.csv", index=False)
    lambdas.to_csv(run_dir / "signed_lambda_diagnostic.csv", index=False)
    synthetic.to_csv(run_dir / "synthetic_cases.csv", index=False)
    synthetic_metrics.to_csv(run_dir / "synthetic_sanity_metrics.csv", index=False)
    plots = render_plots(
        augmented,
        metrics,
        lambdas,
        synthetic,
        synthetic_metrics,
        run_dir / "plots",
    )

    overall = (
        augmented.groupby("model_name", sort=True)
        .apply(direction_metric_row, include_groups=False)
        .apply(pd.Series)
        .reset_index()
    )
    upward_default_models = overall.loc[
        (overall["mean_correction_when_residual_negative"] > 0.0)
        & (overall["mean_correction_when_residual_positive"] > 0.0)
        & (overall["balanced_residual_sign_accuracy"] < 0.55),
        "model_name",
    ].tolist()
    sign_skill_models = overall.loc[
        (overall["balanced_residual_sign_accuracy"] >= 0.55)
        & (overall["correction_residual_correlation"] >= 0.10),
        "model_name",
    ].tolist()
    negative_signed_fits = (
        lambdas.loc[lambdas["best_signed_lambda"] < 0.0, ["model_name", "segment"]]
        .astype(str)
        .agg(" / ".join, axis=1)
        .tolist()
        if not lambdas.empty
        else []
    )
    summary = {
        "status": "residual_direction_audit_complete",
        "test_rows_used": 0,
        "new_quantum_simulation": False,
        "residual_definition": "har_residual = y_true - har_pred",
        "correction_definition": "applied_correction = y_pred - har_pred",
        "synthetic_conclusion": (
            "A signed scalar repairs a globally inverted signal, but it cannot create "
            "sample-specific sign information in an upward-default signal."
        ),
        "upward_default_models": upward_default_models,
        "models_meeting_minimal_sign_skill_rule": sign_skill_models,
        "negative_best_signed_lambda_segments": negative_signed_fits,
        "interpretation": (
            "An upward-default result means the readout/calibration behaves as a risk "
            "overlay rather than a general HAR-residual forecaster. It does not prove "
            "that the reservoir features lack directional information."
        ),
        "plots": plots,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    overall.to_csv(run_dir / "overall_direction_metrics.csv", index=False)
    return run_dir
