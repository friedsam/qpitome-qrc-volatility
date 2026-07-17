#!/usr/bin/env python3
"""Analyze structure of a transition-destination episode run."""
from __future__ import annotations
import argparse
from pathlib import Path
from experiments.runs import begin_run
from transition_destination.core import prepare_binary_episodes, structure_tables

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--neutral-zone", type=float, default=1.0)
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--cluster-gap-weeks", type=int, default=12)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_destination/analyze_structure"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = begin_run(args.results_root, args, run_id=args.run_id, workflow="transition_destination.analyze_structure", return_context=True)
    try:
        run.record_input("episodes", args.episodes)
        frame = prepare_binary_episodes(args.episodes, args.horizon, args.neutral_zone, args.split_date)
        target = f"future_return_{args.horizon}w"
        tables = {"binary_episodes.csv": frame, **{f"{name}.csv": table for name, table in structure_tables(frame, target, args.cluster_gap_weeks).items()}}
        for name, table in tables.items():
            path = run.run_dir / name; table.to_csv(path, index=False); run.record_output(name, path)
        run.finish(); print(run.run_dir)
    except Exception as exc:
        run.finish(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}); raise
if __name__ == "__main__": main()
