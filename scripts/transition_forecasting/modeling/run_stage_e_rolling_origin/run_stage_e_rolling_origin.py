from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.chronological_splits import rolling_origin_assignments
from transition_forecasting.modeling.stage_e_sequence_models import (
    chronology_audit,
    evaluate_fold,
    load_stage_d,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict sample-level rolling-origin Stage E baseline study.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.17)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--input-lookback-days", type=int, default=60)
    parser.add_argument("--target-horizon-days", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/run_stage_e_rolling_origin"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    manifest, sequences = load_stage_d(args.stage_d_run)
    audit_by_year = chronology_audit(manifest, date_column="origin_date")
    assignments, fold_summary, integrity = rolling_origin_assignments(
        manifest,
        n_folds=args.n_folds,
        test_fraction=args.test_fraction,
        embargo_days=args.embargo_days,
        input_lookback_days=args.input_lookback_days,
        target_horizon_days=args.target_horizon_days,
    )

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    audit_by_year.to_csv(run_dir / "chronology_by_origin_year.csv", index=False)
    assignments.to_csv(run_dir / "rolling_fold_manifest.csv", index=False)
    (run_dir / "chronology_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")

    frames = []
    for fold in range(1, args.n_folds + 1):
        fold_manifest = assignments[assignments["fold"].eq(fold)].reset_index(drop=True)
        frames.append(
            evaluate_fold(
                fold_manifest,
                sequences,
                fold=fold,
                seeds=tuple(args.seeds),
            )
        )
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(run_dir / "rolling_results_by_seed.csv", index=False)

    seeded = results[results["seed"] > 0]
    seeded_summary = seeded.groupby(["fold", "model", "alpha"], as_index=False).agg(
        mean_val_qlike=("val_qlike", "mean"),
        std_val_qlike=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
    )
    deterministic = results[results["seed"] == 0].rename(
        columns={"val_qlike": "mean_val_qlike", "val_rmse": "mean_val_rmse"}
    )
    deterministic["std_val_qlike"] = 0.0
    fold_table = pd.concat(
        [
            seeded_summary,
            deterministic[
                ["fold", "model", "alpha", "mean_val_qlike", "std_val_qlike", "mean_val_rmse"]
            ],
        ],
        ignore_index=True,
    )
    fold_table.to_csv(run_dir / "rolling_summary_by_fold.csv", index=False)

    best_per_fold = (
        fold_table.sort_values(["fold", "model", "mean_val_qlike", "mean_val_rmse"])
        .groupby(["fold", "model"], as_index=False)
        .first()
    )
    aggregate = best_per_fold.groupby("model", as_index=False).agg(
        folds=("fold", "nunique"),
        mean_val_qlike=("mean_val_qlike", "mean"),
        std_across_folds=("mean_val_qlike", "std"),
        mean_val_rmse=("mean_val_rmse", "mean"),
        wins=("mean_val_qlike", lambda values: 0),
    )
    winners = best_per_fold.loc[
        best_per_fold.groupby("fold")["mean_val_qlike"].idxmin(), "model"
    ].value_counts()
    aggregate["wins"] = aggregate["model"].map(winners).fillna(0).astype(int)
    aggregate = aggregate.sort_values("mean_val_qlike")
    best_per_fold.to_csv(run_dir / "rolling_best_by_fold.csv", index=False)
    aggregate.to_csv(run_dir / "rolling_aggregate.csv", index=False)

    payload = {
        "stage_d_run": str(args.stage_d_run),
        "test_evaluated": False,
        "chronology_integrity": integrity,
        "folds": fold_summary,
        "aggregate": aggregate.to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
