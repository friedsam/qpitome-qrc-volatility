#!/usr/bin/env python3
"""Create a causal transition-destination target audit run."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from experiments.runs import begin_run
from transition_destination.core import add_future_outcomes, choose_uncertainty_threshold, extract_episodes, load_weekly_predictions, target_summary
from transition_destination.diagnostics import feature_auc

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--reset-threshold", type=float, default=0.35)
    parser.add_argument("--min-gap", type=int, default=6)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--neutral-zones", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--audit-horizon", type=int, default=8)
    parser.add_argument("--audit-neutral-zone", type=float, default=1.0)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_destination/audit_targets"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = begin_run(args.results_root, args, run_id=args.run_id, workflow="transition_destination.audit_targets", return_context=True)
    try:
        run.record_input("predictions", args.predictions)
        frame = load_weekly_predictions(args.predictions).dropna().reset_index(drop=True)
        threshold, split = choose_uncertainty_threshold(frame, args.train_fraction)
        split_date = frame.iloc[split]["date"]
        episodes = add_future_outcomes(extract_episodes(frame, threshold, args.reset_threshold, args.min_gap), frame, args.horizons)
        outputs = {
            "metadata.csv": pd.DataFrame([{"n_weekly_rows": len(frame), "n_episodes": len(episodes), "frozen_uncertainty_threshold": threshold, "split_date": split_date.date()}]),
            "episodes.csv": episodes,
            "target_summary.csv": target_summary(episodes, args.horizons, args.neutral_zones, split_date),
            "feature_auc.csv": feature_auc(episodes, args.audit_horizon, args.audit_neutral_zone, split_date),
        }
        for name, table in outputs.items():
            path = run.run_dir / name
            table.to_csv(path, index=False)
            run.record_output(name, path)
        run.finish()
        print(run.run_dir)
    except Exception as exc:
        run.finish(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        raise
if __name__ == "__main__":
    main()
