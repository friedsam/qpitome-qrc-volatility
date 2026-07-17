#!/usr/bin/env python3
"""Run causal classical destination baselines in an immutable run directory."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from experiments.runs import begin_run
from evaluation.prequential import evaluate_classical, summarize_predictions
from transition_destination.core import prepare_binary_episodes

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--neutral-zone", type=float, default=1.0)
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--minimum-train", type=int, default=60)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_destination/classical_baselines"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = begin_run(args.results_root, args, run_id=args.run_id, workflow="transition_destination.classical_baselines", return_context=True)
    try:
        run.record_input("episodes", args.episodes)
        frame = prepare_binary_episodes(args.episodes, args.horizon, args.neutral_zone)
        predictions, coefficients = evaluate_classical(frame, pd.Timestamp(args.split_date), args.penalty, args.minimum_train)
        if predictions.empty: raise RuntimeError("No predictions produced")
        outputs = {"predictions.csv": predictions, "coefficient_history.csv": coefficients, "metrics.csv": summarize_predictions(predictions, subgroup=True)}
        for name, table in outputs.items():
            path = run.run_dir / name; table.to_csv(path, index=False); run.record_output(name, path)
        run.finish(); print(run.run_dir)
    except Exception as exc:
        run.finish(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}); raise
if __name__ == "__main__": main()
