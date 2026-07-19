from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_fixed_spec_diagnostics import (
    evaluate_fold_fixed_with_mz,
)
from transition_forecasting.modeling.stage_e_sequence_models import load_rematched_rolling

MZ_COLUMNS = (
    "mz_intercept",
    "mz_slope",
    "mz_r2",
    "mz_joint_f",
    "mz_joint_p",
    "mz_horizon_mean_intercept",
    "mz_horizon_mean_slope",
    "mz_horizon_mean_r2",
    "mz_horizon_mean_abs_slope_error",
)


def _aggregate(results: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "val_qlike": ("val_qlike", "mean"),
        "val_rmse": ("val_rmse", "mean"),
        "seed_sd_qlike": ("val_qlike", "std"),
        "seed_sd_rmse": ("val_rmse", "std"),
        "train_samples": ("train_samples", "first"),
        "val_samples": ("val_samples", "first"),
        "seeds": ("seed", "nunique"),
    }
    aggregations.update({column: (column, "mean") for column in MZ_COLUMNS})
    return (
        results.groupby(["fold", "model"], as_index=False)
        .agg(**aggregations)
        .fillna(0.0)
    )


def _pooled(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in per_fold.groupby("model", sort=True):
        weight = group["val_samples"].astype(float)
        row: dict[str, object] = {
            "model": str(model),
            "folds": int(group["fold"].nunique()),
            "mean_fold_qlike": float(group["val_qlike"].mean()),
            "median_fold_qlike": float(group["val_qlike"].median()),
            "weighted_qlike": float((group["val_qlike"] * weight).sum() / weight.sum()),
            "mean_fold_rmse": float(group["val_rmse"].mean()),
            "weighted_rmse": float((group["val_rmse"] * weight).sum() / weight.sum()),
        }
        for column in MZ_COLUMNS:
            row[f"weighted_{column}"] = float((group[column] * weight).sum() / weight.sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("weighted_qlike").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed compact models with RMSE, QLIKE, and MZ diagnostics."
    )
    parser.add_argument("--chronology-run", type=Path, required=True)
    parser.add_argument("--sequence-alpha", type=float, default=100.0)
    parser.add_argument("--esn-alpha", type=float, default=1000.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/transition_forecasting/modeling/run_stage_e_repaired_rolling_origin"
        ),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, tensors = load_rematched_rolling(args.chronology_run)

    rows: list[pd.DataFrame] = []
    for fold in sorted(int(value) for value in manifest["fold"].unique()):
        mask = manifest["fold"].eq(fold) & manifest["fold_split"].isin(["train", "val"])
        fold_manifest = manifest.loc[mask].reset_index(drop=True)
        fold_tensors = tensors[mask.to_numpy()]
        rows.append(
            evaluate_fold_fixed_with_mz(
                fold_manifest,
                fold_tensors,
                fold=fold,
                seeds=tuple(int(seed) for seed in args.seeds),
                sequence_alpha=float(args.sequence_alpha),
                esn_alpha=float(args.esn_alpha),
            )
        )

    raw = pd.concat(rows, ignore_index=True)
    per_fold = _aggregate(raw)
    pooled = _pooled(per_fold)

    raw.to_csv(run_dir / "fixed_spec_seed_results.csv", index=False)
    per_fold.to_csv(run_dir / "fixed_spec_fold_results.csv", index=False)
    pooled.to_csv(run_dir / "fixed_spec_pooled_results.csv", index=False)

    summary = {
        "chronology_run": str(args.chronology_run),
        "folds": sorted(int(value) for value in manifest["fold"].unique()),
        "sequence_alpha": float(args.sequence_alpha),
        "esn_alpha": float(args.esn_alpha),
        "seeds": [int(seed) for seed in args.seeds],
        "models": sorted(raw["model"].unique()),
        "metrics": ["RMSE", "QLIKE", "Mincer-Zarnowitz"],
        "mz_regression": "observed log-volatility = intercept + slope * forecast log-volatility",
        "mz_ideal": {"intercept": 0.0, "slope": 1.0},
        "mz_path_handling": "pooled sample-horizon observations plus mean per-horizon diagnostics",
        "test_rows_used": 0,
        "selection_policy": "fixed specifications; no per-fold validation tuning",
        "best_weighted_qlike_model": str(pooled.iloc[0]["model"]),
        "best_weighted_qlike": float(pooled.iloc[0]["weighted_qlike"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nPooled fixed-spec results:\n")
    display_columns = [
        "model",
        "folds",
        "weighted_qlike",
        "weighted_rmse",
        "weighted_mz_intercept",
        "weighted_mz_slope",
        "weighted_mz_r2",
        "weighted_mz_joint_p",
        "weighted_mz_horizon_mean_abs_slope_error",
    ]
    print(pooled[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
