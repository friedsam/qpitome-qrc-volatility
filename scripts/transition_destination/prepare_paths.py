#!/usr/bin/env python3
"""Prepare immutable causal path tensors for destination experiments."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from experiments.runs import begin_run
from transition_destination.core import DEFAULT_CHANNELS, extract_paths, load_weekly_predictions, prepare_binary_episodes

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly-predictions", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--neutral-zone", type=float, default=1.0)
    parser.add_argument("--path-weeks", type=int, default=13)
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_destination/prepare_paths"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = begin_run(args.results_root, args, run_id=args.run_id, workflow="transition_destination.prepare_paths", return_context=True)
    try:
        run.record_input("weekly_predictions", args.weekly_predictions); run.record_input("episodes", args.episodes)
        weekly = load_weekly_predictions(args.weekly_predictions)
        episodes = prepare_binary_episodes(args.episodes, args.horizon, args.neutral_zone)
        paths, metadata, long, standardization = extract_paths(weekly, episodes, args.channels, args.path_weeks, __import__('pandas').Timestamp(args.split_date))
        np.save(run.run_dir / "paths.npy", paths)
        metadata.to_csv(run.run_dir / "episode_metadata.csv", index=False)
        long.to_csv(run.run_dir / "paths_long.csv", index=False)
        standardization.to_csv(run.run_dir / "standardization.csv", index=False)
        summary = {"array_shape": list(paths.shape), "n_positive": int(metadata["y_positive"].sum()), "n_negative": int(len(metadata)-metadata["y_positive"].sum()), "channels": args.channels}
        (run.run_dir / "dataset_manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
        for name in ("paths.npy", "episode_metadata.csv", "paths_long.csv", "standardization.csv", "dataset_manifest.json"): run.record_output(name, run.run_dir / name)
        run.finish(); print(run.run_dir)
    except Exception as exc:
        run.finish(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}); raise
if __name__ == "__main__": main()
