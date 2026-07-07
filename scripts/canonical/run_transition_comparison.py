#!/usr/bin/env python3
"""Canonical entry point for the Phase 3 transition task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = Path("configs/transition_v1.json")
DEFAULT_OUT_ROOT = Path("results/canonical/transition_v1")
SUPPORTED_MODELS = ("garch", "lstm", "tfim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
    )
    p.add_argument("--only-folds", nargs="*", type=int, default=None)
    p.add_argument("--force-tfim", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    if cfg["target"] != "rv_innovation_20d":
        raise ValueError(f"Unsupported transition target: {cfg['target']}")

    folds = [
        "--n-folds", str(cfg["folds"]),
        "--min-train", str(cfg["min_train"]),
        "--val-size", str(cfg["val_size"]),
        "--purge", str(cfg["purge"]),
    ]
    if args.only_folds:
        folds += ["--only-folds", *map(str, args.only_folds)]

    commands = {
        "garch": [
            sys.executable,
            "scripts/baselines/garch/run_phase3_garch_walkforward.py",
            "--data", cfg["dataset"],
            "--task", "innovation",
            "--out-dir", str(args.out_root / "garch"),
            "--tag", "garch_transition_v1",
            *folds,
        ],
        "lstm": [
            sys.executable,
            "scripts/baselines/lstm/run_phase3_lstm_walkforward.py",
            "--data", cfg["dataset"],
            "--task", "innovation",
            "--lookback", str(cfg["lookback"]),
            "--pca-components", str(cfg["pca_components"]),
            "--out-dir", str(args.out_root / "lstm"),
            "--tag", "lstm_transition_v1",
            *folds,
        ],
        "tfim": [
            sys.executable,
            "scripts/canonical/run_canonical_tfim.py",
            "--data", cfg["dataset"],
            "--task", "innovation",
            "--out-dir", str(args.out_root / "tfim"),
            "--tag", "tfim_transition_v1",
            *folds,
        ],
    }

    if args.force_tfim:
        commands["tfim"].append("--force")

    manifest = {
        "config": str(args.config),
        "task": "innovation",
        "target": cfg["target"],
        "models": args.models,
        "only_folds": args.only_folds,
        "commands": {model: commands[model] for model in args.models},
    }

    for model in args.models:
        command = commands[model]
        print(f"\n=== {model} ===")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)

    if args.dry_run:
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root / "transition_run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
