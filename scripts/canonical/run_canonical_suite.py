#!/usr/bin/env python3
"""Restart-safe orchestration for the full canonical comparison suite.

Each model is an independent segment. Completed segments are skipped on restart.
A failure in one segment does not erase completed work.

Default suite:
- persistence_20d
- har_ridge
- raw_ridge
- esn_selected
- tfim_phase2_final
- rydberg_temporal
- rydberg_memoryless
- rydberg_shuffled
- rydberg_multi_lb
- rydberg_multi_lb_memoryless
- rydberg_multi_lb_shuffled

After all requested segments complete, the script combines per-fold metrics,
predictions, and aggregate summaries into one canonical comparison directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

DEFAULT_MODELS = (
    "persistence_20d",
    "har_ridge",
    "raw_ridge",
    "esn_selected",
    "tfim_phase2_final",
    "rydberg_temporal",
    "rydberg_memoryless",
    "rydberg_shuffled",
    "rydberg_multi_lb",
    "rydberg_multi_lb_memoryless",
    "rydberg_multi_lb_shuffled",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p.add_argument("--tag", default="phase3_current")
    p.add_argument("--out-dir", type=Path, default=Path("results/canonical/current"))
    p.add_argument("--segments-dir", type=Path, default=Path("results/canonical/segments"))
    p.add_argument("--only-folds", nargs="*", type=int, default=None)
    p.add_argument("--protocols", nargs="*", default=["exact"])
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--shot-seeds", default="7")
    p.add_argument("--force", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def segment_paths(root: Path, model: str, tag: str) -> dict[str, Path]:
    seg = root / model
    return {
        "dir": seg,
        "metrics": seg / f"per_fold_metrics_{tag}_{model}.csv",
        "predictions": seg / f"predictions_{tag}_{model}.csv",
        "aggregate": seg / f"aggregate_metrics_{tag}_{model}.csv",
        "manifest": seg / f"run_manifest_{tag}_{model}.json",
    }


def is_complete(paths: dict[str, Path]) -> bool:
    return all(paths[name].exists() for name in ("metrics", "predictions", "aggregate", "manifest"))


def run_segment(args: argparse.Namespace, model: str) -> bool:
    tag = f"{args.tag}_{model}"
    paths = segment_paths(args.segments_dir, model, args.tag)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if is_complete(paths) and not args.force:
        print(f"SKIP complete segment: {model}")
        return True

    if model == "har_ridge":
        cmd = [
            sys.executable,
            "scripts/canonical/run_canonical_har.py",
            "--out-dir", str(paths["dir"]),
            "--tag", tag,
        ]
        if args.only_folds:
            cmd += ["--only-folds", *[str(x) for x in args.only_folds]]
        if args.force:
            cmd.append("--force")
    elif model == "tfim_phase2_final":
        cmd = [
            sys.executable,
            "scripts/canonical/run_canonical_tfim.py",
            "--out-dir", str(paths["dir"]),
            "--tag", tag,
        ]
        if args.only_folds:
            cmd += ["--only-folds", *[str(x) for x in args.only_folds]]
        if args.force:
            cmd.append("--force")
    else:
        cmd = [
            sys.executable,
            "scripts/canonical/run_canonical_comparison.py",
            "--models", model,
            "--out-dir", str(paths["dir"]),
            "--tag", tag,
            "--protocols", *args.protocols,
            "--shots", str(args.shots),
            "--shot-seeds", args.shot_seeds,
        ]
        if args.only_folds:
            cmd += ["--only-folds", *[str(x) for x in args.only_folds]]
        if args.force:
            cmd.append("--force-recompute")

    print(f"\nRUN segment: {model}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"FAILED segment: {model} (exit {result.returncode})")
        return False

    if not is_complete(paths):
        print(f"FAILED completeness check: {model}")
        return False

    print(f"DONE segment: {model}")
    return True


def combine_completed(args: argparse.Namespace) -> None:
    metrics_frames = []
    prediction_frames = []
    aggregate_frames = []
    completed = []
    missing = []

    for model in args.models:
        paths = segment_paths(args.segments_dir, model, args.tag)
        if not is_complete(paths):
            missing.append(model)
            continue
        completed.append(model)
        metrics_frames.append(pd.read_csv(paths["metrics"]))
        prediction_frames.append(pd.read_csv(paths["predictions"]))
        aggregate_frames.append(pd.read_csv(paths["aggregate"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_frames:
        pd.concat(metrics_frames, ignore_index=True).to_csv(
            args.out_dir / f"per_fold_metrics_{args.tag}.csv", index=False
        )
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            args.out_dir / f"predictions_{args.tag}.csv", index=False
        )
    if aggregate_frames:
        pd.concat(aggregate_frames, ignore_index=True).to_csv(
            args.out_dir / f"aggregate_metrics_{args.tag}.csv", index=False
        )

    manifest = {
        "tag": args.tag,
        "requested_models": args.models,
        "completed_models": completed,
        "missing_models": missing,
        "protocols": args.protocols,
        "only_folds": args.only_folds,
        "restart_policy": "each model is an independent segment; complete segments are skipped unless --force; TFIM also resumes by completed fold",
    }
    (args.out_dir / f"run_manifest_{args.tag}.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nCombined {len(completed)} completed segments; missing={missing}")


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    failures = []
    for model in args.models:
        ok = run_segment(args, model)
        if not ok:
            failures.append(model)
            if not args.continue_on_error:
                combine_completed(args)
                raise SystemExit(f"Stopped after failed segment: {model}")

    combine_completed(args)
    if failures:
        raise SystemExit(f"Suite completed with failed segments: {failures}")


if __name__ == "__main__":
    main()
