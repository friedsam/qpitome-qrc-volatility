from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN = {
    "pls5_s0.5": {"components": 5, "shrinkage": 0.5},
    "pls10_s0.75": {"components": 10, "shrinkage": 0.75},
}


def weighted(group: pd.DataFrame, column: str) -> float:
    weight = group["val_samples"].astype(float)
    return float((group[column] * weight).sum() / weight.sum())


def load_pls(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "fold",
        "order",
        "components",
        "shrinkage",
        "val_samples",
        "val_qlike",
        "val_rmse",
        "mz_r2",
        "mz_horizon_mean_abs_slope_error",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def select_frozen(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, spec in FROZEN.items():
        chosen = frame[
            frame["components"].eq(spec["components"])
            & np.isclose(frame["shrinkage"], spec["shrinkage"])
        ].copy()
        if chosen.empty:
            raise ValueError(f"missing frozen specification {name}")
        chosen["candidate"] = name
        rows.append(chosen)
    return pd.concat(rows, ignore_index=True)


def load_har(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"fold", "model", "val_samples", "val_qlike", "val_rmse", "mz_r2", "mz_horizon_mean_abs_slope_error"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    har = frame[frame["model"].eq("har")].copy()
    if har["fold"].nunique() != 8:
        raise ValueError("HAR file must contain all eight folds")
    return har


def make_fold_table(pls: pd.DataFrame, har: pd.DataFrame) -> pd.DataFrame:
    ordered = pls[pls["order"].eq("ordered")].copy()
    shuffled = pls[pls["order"].eq("shuffled")].copy()
    key = ["fold", "candidate"]
    comparison = ordered.merge(
        shuffled,
        on=key,
        suffixes=("_ordered", "_shuffled"),
        validate="one_to_one",
    )
    har_small = har[["fold", "val_qlike", "val_rmse", "mz_r2", "mz_horizon_mean_abs_slope_error"]].rename(
        columns={
            "val_qlike": "val_qlike_har",
            "val_rmse": "val_rmse_har",
            "mz_r2": "mz_r2_har",
            "mz_horizon_mean_abs_slope_error": "mz_horizon_mean_abs_slope_error_har",
        }
    )
    comparison = comparison.merge(har_small, on="fold", how="left", validate="many_to_one")
    comparison["val_samples"] = comparison["val_samples_ordered"]
    for metric in ("val_qlike", "val_rmse", "mz_r2", "mz_horizon_mean_abs_slope_error"):
        comparison[f"ordered_minus_har_{metric}"] = comparison[f"{metric}_ordered"] - comparison[f"{metric}_har"]
        comparison[f"ordered_minus_shuffled_{metric}"] = comparison[f"{metric}_ordered"] - comparison[f"{metric}_shuffled"]
    return comparison.sort_values(["candidate", "fold"]).reset_index(drop=True)


def summarize_subset(group: pd.DataFrame, label: str) -> dict[str, object]:
    result: dict[str, object] = {
        "subset": label,
        "folds": ",".join(str(int(value)) for value in sorted(group["fold"].unique())),
        "n_folds": int(group["fold"].nunique()),
    }
    for target in ("har", "shuffled"):
        for metric in ("val_qlike", "val_rmse", "mz_r2", "mz_horizon_mean_abs_slope_error"):
            column = f"ordered_minus_{target}_{metric}"
            result[f"weighted_{column}"] = weighted(group, column)
            result[f"median_{column}"] = float(group[column].median())
        result[f"qlike_wins_vs_{target}"] = int((group[f"ordered_minus_{target}_val_qlike"] < 0).sum())
        result[f"rmse_wins_vs_{target}"] = int((group[f"ordered_minus_{target}_val_rmse"] < 0).sum())
    return result


def summarize_stability(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    for candidate, group in table.groupby("candidate", sort=True):
        all_row = summarize_subset(group, "all_8")
        all_row["candidate"] = candidate
        summary_rows.append(all_row)

        no6 = group[~group["fold"].eq(6)]
        no6_row = summarize_subset(no6, "exclude_fold_6")
        no6_row["candidate"] = candidate
        summary_rows.append(no6_row)

        for omitted in sorted(int(value) for value in group["fold"].unique()):
            subset = group[~group["fold"].eq(omitted)]
            row = summarize_subset(subset, f"omit_{omitted}")
            row["candidate"] = candidate
            row["omitted_fold"] = omitted
            loo_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    loo = pd.DataFrame(loo_rows)

    range_rows: list[dict[str, object]] = []
    delta_columns = [
        column for column in loo.columns if column.startswith("weighted_ordered_minus_")
    ]
    for candidate, group in loo.groupby("candidate", sort=True):
        row: dict[str, object] = {"candidate": candidate}
        for column in delta_columns:
            row[f"{column}_loo_min"] = float(group[column].min())
            row[f"{column}_loo_max"] = float(group[column].max())
            row[f"{column}_sign_stable"] = bool(
                (group[column] < 0).all() or (group[column] > 0).all()
            )
        range_rows.append(row)
    ranges = pd.DataFrame(range_rows)
    return summary, loo.merge(ranges, on="candidate", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact stability analysis for frozen Stage E models.")
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--secondary-run", type=Path, required=True)
    parser.add_argument("--fixed-spec-run", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/analyze_stage_e_fold_stability"),
    )
    parser.add_argument("--run-id", default="frozen_models_8fold_001")
    args = parser.parse_args()

    run_dir = args.out_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    development = load_pls(args.development_run / "pls_fold_results.csv")
    secondary = load_pls(args.secondary_run / "pls_fold_results.csv")
    pls = select_frozen(pd.concat([development, secondary], ignore_index=True))
    if sorted(int(value) for value in pls["fold"].unique()) != list(range(1, 9)):
        raise ValueError("combined PLS inputs must cover folds 1 through 8")

    har = load_har(args.fixed_spec_run / "fixed_spec_fold_results.csv")
    fold_table = make_fold_table(pls, har)
    summary, loo = summarize_stability(fold_table)

    fold_table.to_csv(run_dir / "fold_level_differences.csv", index=False)
    summary.to_csv(run_dir / "all_and_exclude_fold6_summary.csv", index=False)
    loo.to_csv(run_dir / "leave_one_fold_out_summary.csv", index=False)

    compact_columns = [
        "candidate",
        "subset",
        "n_folds",
        "weighted_ordered_minus_har_val_qlike",
        "median_ordered_minus_har_val_qlike",
        "qlike_wins_vs_har",
        "weighted_ordered_minus_har_val_rmse",
        "rmse_wins_vs_har",
        "weighted_ordered_minus_shuffled_val_qlike",
        "qlike_wins_vs_shuffled",
        "weighted_ordered_minus_shuffled_val_rmse",
        "rmse_wins_vs_shuffled",
    ]
    print("\nAll folds and fold-6 sensitivity:\n")
    print(summary[compact_columns].to_string(index=False))

    range_columns = [
        "candidate",
        "weighted_ordered_minus_har_val_qlike_loo_min",
        "weighted_ordered_minus_har_val_qlike_loo_max",
        "weighted_ordered_minus_har_val_qlike_sign_stable",
        "weighted_ordered_minus_shuffled_val_qlike_loo_min",
        "weighted_ordered_minus_shuffled_val_qlike_loo_max",
        "weighted_ordered_minus_shuffled_val_qlike_sign_stable",
        "weighted_ordered_minus_har_val_rmse_loo_min",
        "weighted_ordered_minus_har_val_rmse_loo_max",
        "weighted_ordered_minus_har_val_rmse_sign_stable",
        "weighted_ordered_minus_shuffled_val_rmse_loo_min",
        "weighted_ordered_minus_shuffled_val_rmse_loo_max",
        "weighted_ordered_minus_shuffled_val_rmse_sign_stable",
    ]
    ranges = loo[range_columns].drop_duplicates().sort_values("candidate")
    print("\nLeave-one-fold-out ranges:\n")
    print(ranges.to_string(index=False))

    metadata = {
        "purpose": "predeclared compact fold stability analysis",
        "frozen_models": FROZEN,
        "primary_outputs": [
            "individual fold differences",
            "all-eight-fold summary",
            "fold-6-excluded sensitivity",
            "leave-one-fold-out ranges",
        ],
        "no_new_model_selection": True,
        "test_rows_used": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
