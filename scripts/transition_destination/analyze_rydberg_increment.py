#!/usr/bin/env python3
"""Run paired diagnostics for a protected baseline and Rydberg candidate."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from evaluation.paired import compare_predictions
from experiments.runs import begin_run

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-model", default="momentum_13w")
    parser.add_argument("--candidate-model", default="momentum_plus_rydberg_return_uncertainty")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_destination/analyze_rydberg_increment"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = begin_run(args.results_root, args, run_id=args.run_id, workflow="transition_destination.analyze_rydberg_increment", return_context=True)
    try:
        run.record_input("predictions", args.predictions)
        frame = pd.read_csv(args.predictions); frame["date"] = pd.to_datetime(frame["date"])
        paired, summary, thresholds, correction = compare_predictions(frame, args.baseline_model, args.candidate_model, args.n_bootstrap, args.seed)
        if args.expected_episodes is not None and len(paired) != args.expected_episodes:
            raise ValueError(f"Expected {args.expected_episodes} paired episodes, found {len(paired)}")
        outputs = {"paired_episode_comparison.csv": paired, "paired_loss_summary.csv": summary, "threshold_tradeoff.csv": thresholds, "class_probability_shift.csv": correction}
        for name, table in outputs.items():
            path = run.run_dir / name; table.to_csv(path, index=False); run.record_output(name, path)
        run.finish(); print(run.run_dir)
    except Exception as exc:
        run.finish(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}); raise
if __name__ == "__main__": main()
