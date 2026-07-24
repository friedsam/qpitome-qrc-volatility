from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run


@dataclass(frozen=True)
class QlikeDominanceConfig:
    """Post-process the residual-head root-cause assay without new simulation."""

    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    lambda_cap: float = 1.25
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25)
    exponential_clip: float = 50.0
    correction_tolerance: float = 1e-12

    def validate(self) -> None:
        if not self.lambda_grid or 0.0 not in self.lambda_grid:
            raise ValueError("lambda_grid must be nonempty and contain zero")
        if any(float(value) < 0.0 for value in self.lambda_grid):
            raise ValueError("lambda_grid must be nonnegative")
        if self.lambda_cap <= 0.0:
            raise ValueError("lambda_cap must be positive")
        if max(self.lambda_grid) > self.lambda_cap:
            raise ValueError("lambda_grid cannot exceed lambda_cap")
        if not self.top_fractions or any(
            not 0.0 < float(value) <= 1.0 for value in self.top_fractions
        ):
            raise ValueError("top_fractions must lie in (0, 1]")
        if self.exponential_clip <= 0.0:
            raise ValueError("exponential_clip must be positive")
        if self.correction_tolerance <= 0.0:
            raise ValueError("correction_tolerance must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def qlike_cell_loss(
    residual: np.ndarray,
    correction: np.ndarray,
    lambda_value: float | np.ndarray,
    *,
    exponential_clip: float = 50.0,
) -> np.ndarray:
    """QLIKE cell loss for log-volatility residual r=y-HAR and HAR+lambda*c."""

    r = np.asarray(residual, dtype=float)
    c = np.asarray(correction, dtype=float)
    value = np.asarray(lambda_value, dtype=float)
    difference = 2.0 * (r - value * c)
    clipped = np.clip(difference, -float(exponential_clip), float(exponential_clip))
    return np.exp(clipped) - difference - 1.0


def qlike_gradient_wrt_lambda(
    residual: np.ndarray,
    correction: np.ndarray,
    lambda_value: float | np.ndarray,
    *,
    exponential_clip: float = 50.0,
) -> np.ndarray:
    """Derivative of QLIKE with respect to a nonnegative correction multiplier."""

    r = np.asarray(residual, dtype=float)
    c = np.asarray(correction, dtype=float)
    value = np.asarray(lambda_value, dtype=float)
    difference = 2.0 * (r - value * c)
    clipped = np.clip(difference, -float(exponential_clip), float(exponential_clip))
    return 2.0 * c * (1.0 - np.exp(clipped))


def bounded_cellwise_optimal_lambda(
    residual: np.ndarray,
    correction: np.ndarray,
    *,
    lambda_cap: float,
    correction_tolerance: float = 1e-12,
) -> np.ndarray:
    """Cellwise QLIKE optimum under 0 <= lambda <= lambda_cap.

    If the correction has the wrong sign, the optimum is zero. If the signs agree,
    the unconstrained optimum is residual/correction and is clipped at lambda_cap.
    """

    r = np.asarray(residual, dtype=float)
    c = np.asarray(correction, dtype=float)
    if r.shape != c.shape:
        raise ValueError("residual and correction must have the same shape")
    output = np.zeros_like(r, dtype=float)
    usable = (np.abs(c) > float(correction_tolerance)) & ((r * c) > 0.0)
    output[usable] = np.clip(r[usable] / c[usable], 0.0, float(lambda_cap))
    return output


def _sign(values: np.ndarray, tolerance: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(
        array > tolerance,
        "positive",
        np.where(array < -tolerance, "negative", "zero"),
    )


def _validate_cells(cells: pd.DataFrame) -> None:
    required = {
        "fold",
        "model_family",
        "readout",
        "population",
        "segment",
        "har_residual",
        "raw_correction",
    }
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(f"raw correction cells are missing columns: {sorted(missing)}")
    if cells.empty:
        raise ValueError("raw correction cells are empty")


def _validate_selections(selections: pd.DataFrame) -> None:
    required = {
        "fold",
        "model_family",
        "readout",
        "selected_early_lambda",
        "selected_transition_lambda",
    }
    missing = required.difference(selections.columns)
    if missing:
        raise ValueError(f"selected lambdas are missing columns: {sorted(missing)}")
    if selections.empty:
        raise ValueError("selected lambdas are empty")


def annotate_selected_lambdas(
    cells: pd.DataFrame,
    selections: pd.DataFrame,
    config: QlikeDominanceConfig,
) -> pd.DataFrame:
    """Attach the selected segment lambda and per-cell QLIKE diagnostics."""

    config.validate()
    _validate_cells(cells)
    _validate_selections(selections)
    keys = ["fold", "model_family", "readout"]
    selected = selections[
        keys + ["selected_early_lambda", "selected_transition_lambda"]
    ].drop_duplicates(keys)
    if selected.duplicated(keys).any():
        raise ValueError("selected lambda rows are not unique")
    output = cells.merge(selected, on=keys, how="left", validate="many_to_one")
    if output["selected_early_lambda"].isna().any():
        raise ValueError("some correction cells have no selected lambda")
    output["selected_lambda"] = np.where(
        output["segment"].eq("early"),
        output["selected_early_lambda"],
        output["selected_transition_lambda"],
    ).astype(float)

    residual = output["har_residual"].to_numpy(dtype=float)
    correction = output["raw_correction"].to_numpy(dtype=float)
    selected_lambda = output["selected_lambda"].to_numpy(dtype=float)
    output["residual_sign"] = _sign(residual, config.correction_tolerance)
    output["correction_sign"] = _sign(correction, config.correction_tolerance)
    output["sign_match"] = (
        (output["residual_sign"] == output["correction_sign"])
        & ~output["residual_sign"].eq("zero")
    )
    output["baseline_qlike"] = qlike_cell_loss(
        residual,
        correction,
        0.0,
        exponential_clip=config.exponential_clip,
    )
    output["selected_qlike"] = qlike_cell_loss(
        residual,
        correction,
        selected_lambda,
        exponential_clip=config.exponential_clip,
    )
    output["qlike_gain"] = output["baseline_qlike"] - output["selected_qlike"]
    output["gradient_at_zero"] = qlike_gradient_wrt_lambda(
        residual,
        correction,
        0.0,
        exponential_clip=config.exponential_clip,
    )
    output["cellwise_optimal_lambda"] = bounded_cellwise_optimal_lambda(
        residual,
        correction,
        lambda_cap=config.lambda_cap,
        correction_tolerance=config.correction_tolerance,
    )
    output["cellwise_optimum_hits_cap"] = np.isclose(
        output["cellwise_optimal_lambda"], config.lambda_cap
    ) & output["sign_match"]
    output["selected_hits_cap"] = np.isclose(selected_lambda, config.lambda_cap)
    return output


def lambda_grid_summary(
    cells: pd.DataFrame,
    config: QlikeDominanceConfig,
) -> pd.DataFrame:
    """Compare QLIKE and RMSE choices on the same raw correction cells."""

    config.validate()
    _validate_cells(cells)
    keys = ["fold", "model_family", "readout", "population", "segment"]
    rows: list[dict[str, object]] = []
    for values, group in cells.groupby(keys, sort=True):
        residual = group["har_residual"].to_numpy(dtype=float)
        correction = group["raw_correction"].to_numpy(dtype=float)
        for value in config.lambda_grid:
            errors = residual - float(value) * correction
            losses = qlike_cell_loss(
                residual,
                correction,
                float(value),
                exponential_clip=config.exponential_clip,
            )
            rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "lambda": float(value),
                    "cells": int(len(group)),
                    "qlike": float(np.mean(losses)),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "sum_gradient": float(
                        np.sum(
                            qlike_gradient_wrt_lambda(
                                residual,
                                correction,
                                float(value),
                                exponential_clip=config.exponential_clip,
                            )
                        )
                    ),
                    "wrong_sign_cells": int(((residual * correction) <= 0.0).sum()),
                    "wrong_sign_fraction": float(((residual * correction) <= 0.0).mean()),
                }
            )
    output = pd.DataFrame(rows)
    qlike_best = output.loc[
        output.groupby(keys)["qlike"].idxmin(), keys + ["lambda"]
    ].rename(columns={"lambda": "qlike_selected_lambda"})
    rmse_best = output.loc[
        output.groupby(keys)["rmse"].idxmin(), keys + ["lambda"]
    ].rename(columns={"lambda": "rmse_selected_lambda"})
    return output.merge(qlike_best, on=keys, validate="many_to_one").merge(
        rmse_best, on=keys, validate="many_to_one"
    )


def qlike_pressure_summary(annotated: pd.DataFrame) -> pd.DataFrame:
    """Show which residual/sign groups create aggregate pressure for lambda > 0."""

    keys = [
        "model_family",
        "readout",
        "population",
        "segment",
        "residual_sign",
        "correction_sign",
        "sign_match",
    ]
    rows: list[dict[str, object]] = []
    total_positive_gain = annotated.groupby(
        ["model_family", "readout", "population", "segment"], sort=True
    )["qlike_gain"].apply(lambda values: float(np.clip(values, 0.0, None).sum()))
    for values, group in annotated.groupby(keys, sort=True):
        parent = tuple(values[:4])
        positive_gain = float(np.clip(group["qlike_gain"], 0.0, None).sum())
        denominator = float(total_positive_gain.loc[parent])
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "cells": int(len(group)),
                "mean_har_residual": float(group["har_residual"].mean()),
                "mean_raw_correction": float(group["raw_correction"].mean()),
                "mean_baseline_qlike": float(group["baseline_qlike"].mean()),
                "net_qlike_gain": float(group["qlike_gain"].sum()),
                "positive_qlike_gain": positive_gain,
                "share_of_positive_gain": (
                    positive_gain / denominator if denominator > 0.0 else float("nan")
                ),
                "sum_gradient_at_zero": float(group["gradient_at_zero"].sum()),
                "mean_cellwise_optimal_lambda": float(
                    group["cellwise_optimal_lambda"].mean()
                ),
                "cellwise_cap_rate": float(group["cellwise_optimum_hits_cap"].mean()),
            }
        )
    return pd.DataFrame(rows)


def qlike_gain_concentration(
    annotated: pd.DataFrame,
    config: QlikeDominanceConfig,
) -> pd.DataFrame:
    """Measure whether a small set of cells explains the selected-lambda QLIKE gain."""

    keys = ["model_family", "readout", "population", "segment"]
    rows: list[dict[str, object]] = []
    for values, group in annotated.groupby(keys, sort=True):
        gains = np.sort(np.clip(group["qlike_gain"].to_numpy(dtype=float), 0.0, None))[::-1]
        total = float(gains.sum())
        for fraction in config.top_fractions:
            count = max(1, int(np.ceil(float(fraction) * len(gains))))
            captured = float(gains[:count].sum())
            rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "top_fraction": float(fraction),
                    "top_cells": int(count),
                    "cells": int(len(gains)),
                    "positive_gain_total": total,
                    "positive_gain_captured": captured,
                    "share_of_positive_gain": (
                        captured / total if total > 0.0 else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_residual_head_qlike_dominance(
    *,
    root_cause_run: Path,
    results_root: Path,
    config: QlikeDominanceConfig = QlikeDominanceConfig(),
    run_id: str | None = None,
) -> Path:
    """Analyze an existing root-cause assay; no QRC simulation or test rows."""

    config.validate()
    root = Path(root_cause_run)
    cells_path = root / "raw_correction_cells.csv.gz"
    selection_path = root / "selected_lambdas_by_fold.csv"
    if not cells_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(
            "root-cause run must contain raw_correction_cells.csv.gz and "
            "selected_lambdas_by_fold.csv"
        )
    cells = pd.read_csv(cells_path)
    selections = pd.read_csv(selection_path)
    annotated = annotate_selected_lambdas(cells, selections, config)
    grid = lambda_grid_summary(cells, config)
    pressure = qlike_pressure_summary(annotated)
    concentration = qlike_gain_concentration(annotated, config)

    run_dir = begin_run(
        Path(results_root),
        {
            "root_cause_run": str(root),
            "config": config.to_dict(),
            "scientific_question": (
                "Does exponential QLIKE pressure from a small set of positive HAR residuals "
                "select a positive/capped lambda despite wrong-sign corrections elsewhere?"
            ),
            "new_quantum_simulation": False,
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )
    annotated.to_csv(
        run_dir / "qlike_cell_diagnostics.csv.gz", index=False, compression="gzip"
    )
    grid.to_csv(run_dir / "lambda_grid_qlike_vs_rmse.csv", index=False)
    pressure.to_csv(run_dir / "qlike_pressure_by_sign.csv", index=False)
    concentration.to_csv(run_dir / "qlike_gain_concentration.csv", index=False)

    validation = annotated.loc[annotated["population"].eq("validation")]
    summary = {
        "status": "residual_head_qlike_dominance_complete",
        "test_rows_used": 0,
        "new_quantum_simulation": False,
        "validation_cells": int(len(validation)),
        "validation_wrong_sign_fraction": float((~validation["sign_match"]).mean()),
        "validation_cellwise_cap_rate": float(
            validation["cellwise_optimum_hits_cap"].mean()
        ),
        "validation_net_qlike_gain": float(validation["qlike_gain"].sum()),
        "interpretation_rule": (
            "A negative aggregate derivative at lambda=0 means QLIKE pushes lambda upward. "
            "If that pressure and most positive gain come from a small positive-residual tail, "
            "while wrong-sign rows have cellwise optimum zero, pooled QLIKE is selecting a "
            "tail-protection uplift rather than a generally signed residual correction."
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
