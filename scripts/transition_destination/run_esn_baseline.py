#!/usr/bin/env python3
"""Run the matched ESN destination baseline from a prepared path run."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from baselines.destination_esn import make_reservoir, reservoir_features
from experiments.runs import begin_run
from evaluation.prequential import evaluate_reservoir_features, summarize_predictions

SEED = 20260712

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-run", type=Path, required=True)
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--n-units", type=int, default=24)
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--input-scale", type=float, default=0.35)
    parser.add_argument("--leak", type=float, default=0.5)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--momentum-penalty", type=float, default=1.0)
    parser.add_argument("--minimum-train", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_destination/esn_baseline"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = begin_run(args.results_root, args, run_id=args.run_id, workflow="transition_destination.esn_baseline", return_context=True)
    try:
        paths_path, metadata_path = args.path_run / "paths.npy", args.path_run / "episode_metadata.csv"
        run.record_input("paths", paths_path); run.record_input("metadata", metadata_path)
        paths = np.load(paths_path); metadata = pd.read_csv(metadata_path)
        metadata["date"] = pd.to_datetime(metadata["date"]); metadata["outcome_available_date"] = pd.to_datetime(metadata["outcome_available_date"])
        w_in, w, bias = make_reservoir(paths.shape[2], args.n_units, args.spectral_radius, args.input_scale, args.seed)
        features = reservoir_features(paths, w_in, w, bias, args.leak)
        predictions = evaluate_reservoir_features({"direct": features}, metadata, pd.Timestamp(args.split_date), args.minimum_train, args.penalty, args.momentum_penalty, "esn")
        np.save(run.run_dir / "reservoir_features.npy", features)
        predictions.to_csv(run.run_dir / "predictions.csv", index=False)
        summarize_predictions(predictions).to_csv(run.run_dir / "metrics.csv", index=False)
        for name in ("reservoir_features.npy", "predictions.csv", "metrics.csv"): run.record_output(name, run.run_dir / name)
        run.finish(); print(run.run_dir)
    except Exception as exc:
        run.finish(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}); raise
if __name__ == "__main__": main()
