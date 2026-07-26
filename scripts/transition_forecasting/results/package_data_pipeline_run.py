#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.results.data_pipeline_package import (
    package_data_pipeline_run,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a successful transition-data run into a compact evidence package."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/runs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/transition_forecasting/data_pipeline/run"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("docs/transition_forecasting/result_index.csv"),
    )
    parser.add_argument(
        "--experiment-name",
        default="global-transition-data-pipeline-fallback-verified",
    )
    parser.add_argument("--platform", default="qBraid Lab")
    return parser.parse_args()


def main() -> int:
    if Path.cwd().resolve() != REPO_ROOT.resolve():
        raise RuntimeError(
            f"run this command from the repository root: {REPO_ROOT}"
        )
    args = parse_args()
    report = package_data_pipeline_run(
        run_root=args.runs_root / args.run_id,
        output_root=args.output_root,
        index_path=args.index,
        experiment_name=args.experiment_name,
        platform_name=args.platform,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
