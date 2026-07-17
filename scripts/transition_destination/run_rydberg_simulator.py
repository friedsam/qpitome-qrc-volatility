#!/usr/bin/env python3
"""Run the exact-state temporal Rydberg destination reservoir."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from experiments.runs import begin_run
from evaluation.prequential import evaluate_reservoir_features, summarize_predictions
from reservoirs.destination_rydberg import compute_feature_sets

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--encodings", nargs="+", default=["return_only", "return_uncertainty"])
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--minimum-train", type=int, default=60)
    parser.add_argument("--n-atoms", type=int, default=6)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--detuning-base", type=float, default=0.0)
    parser.add_argument("--detuning-scale", type=float, default=1.5)
    parser.add_argument("--interaction-strength", type=float, default=1.2)
    parser.add_argument("--segment-time", type=float, default=0.35)
    parser.add_argument("--quantization-levels", type=int, default=41)
    parser.add_argument("--readout-penalty", type=float, default=10.0)
    parser.add_argument("--momentum-penalty", type=float, default=1.0)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_destination/rydberg_simulator"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = begin_run(args.results_root, args, run_id=args.run_id, workflow="transition_destination.rydberg_simulator", return_context=True)
    try:
        metadata_path = args.input_run / "episode_metadata.csv"; run.record_input("metadata", metadata_path)
        metadata = pd.read_csv(metadata_path); metadata["date"] = pd.to_datetime(metadata["date"]); metadata["outcome_available_date"] = pd.to_datetime(metadata["outcome_available_date"])
        encodings = {}
        for name in args.encodings:
            path = args.input_run / f"{name}.npy"; run.record_input(name, path); encodings[name] = np.load(path)
        features = compute_feature_sets(encodings, n_atoms=args.n_atoms, omega=args.omega, detuning_base=args.detuning_base, detuning_scale=args.detuning_scale, interaction_strength=args.interaction_strength, segment_time=args.segment_time, quantization_levels=args.quantization_levels)
        predictions = evaluate_reservoir_features(features, metadata, pd.Timestamp(args.split_date), args.minimum_train, args.readout_penalty, args.momentum_penalty, "rydberg")
        predictions.to_csv(run.run_dir / "predictions.csv", index=False); summarize_predictions(predictions).to_csv(run.run_dir / "metrics.csv", index=False)
        for name, array in features.items(): np.save(run.run_dir / f"features_{name}.npy", array)
        for path in run.run_dir.iterdir():
            if path.is_file() and path.name not in {"params.json", "run_manifest.json"}: run.record_output(path.name, path)
        run.finish(); print(run.run_dir)
    except Exception as exc:
        run.finish(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}); raise
if __name__ == "__main__": main()
