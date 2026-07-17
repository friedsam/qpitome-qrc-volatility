#!/usr/bin/env python3
"""Build persistent-volatility transition events and matched controls."""
from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.catalogue.transition_events import run_pipeline
from experiments.runs import begin_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/build_transition_events"),
        help="Script-level results root; begin_run creates the final run-id directory.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    summary = run_pipeline(args.data_dir, run_dir, args.seed)
    print(f"Run directory: {run_dir}")
    print(f"Events: {summary['events']}")
    print(f"Global episodes: {summary['global_episodes']}")


if __name__ == "__main__":
    main()
