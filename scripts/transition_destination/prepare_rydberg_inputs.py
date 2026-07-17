#!/usr/bin/env python3
"""Prepare bounded Rydberg control encodings from a prepared path run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from experiments.runs import begin_run
from transition_destination.core import encode_rydberg_inputs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-run", type=Path, required=True)
    parser.add_argument("--bound-divisor", type=float, default=2.0)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_destination/prepare_rydberg_inputs"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = begin_run(args.results_root, args, run_id=args.run_id, workflow="transition_destination.prepare_rydberg_inputs", return_context=True)
    try:
        paths_path = args.path_run / "paths.npy"; metadata_path = args.path_run / "episode_metadata.csv"; standardization_path = args.path_run / "standardization.csv"
        for name, path in (("paths", paths_path), ("metadata", metadata_path), ("standardization", standardization_path)): run.record_input(name, path)
        paths = np.load(paths_path); metadata = pd.read_csv(metadata_path); standardization = pd.read_csv(standardization_path)
        encodings, summary = encode_rydberg_inputs(paths, standardization, args.bound_divisor)
        for name, tensor in encodings.items(): np.save(run.run_dir / f"{name}.npy", tensor)
        metadata.to_csv(run.run_dir / "episode_metadata.csv", index=False)
        summary.to_csv(run.run_dir / "encoding_summary.csv", index=False)
        manifest = {name: {"shape": list(tensor.shape)} for name, tensor in encodings.items()}
        (run.run_dir / "encoding_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        for path in run.run_dir.iterdir():
            if path.is_file() and path.name not in {"params.json", "run_manifest.json"}: run.record_output(path.name, path)
        run.finish(); print(run.run_dir)
    except Exception as exc:
        run.finish(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}); raise
if __name__ == "__main__": main()
