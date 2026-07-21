from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_command(argv: list[str]) -> None:
    completed = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build transition-forecasting data under the regular domain results tree."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/transition_forecasting/data/transition-data-classical-001"),
    )
    parser.add_argument(
        "--source-mode",
        choices=("auto", "live", "fallback"),
        default="live",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.output_dir
    raw = root / "raw" / "global_stock_indices_historical_data"
    processed = root / "processed"
    dataset_1d = processed / "global_transition_dataset_1d"
    dataset_3d = processed / "global_transition_dataset_3d"
    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    overwrite = ["--force"] if args.force else []

    commands = [
        [
            sys.executable,
            "scripts/transition_forecasting/data/acquire_global_index_data.py",
            "--destination",
            str(raw),
            "--source-mode",
            args.source_mode,
            *overwrite,
        ],
        [
            sys.executable,
            "scripts/transition_forecasting/data/build_transition_datasets.py",
            "--raw-root",
            str(raw),
            "--output-1d",
            str(dataset_1d),
            "--output-3d",
            str(dataset_3d),
            *overwrite,
        ],
        [
            sys.executable,
            "scripts/transition_forecasting/data/build_transition_folds.py",
            "--dataset-1d",
            str(dataset_1d),
            "--dataset-3d",
            str(dataset_3d),
            *overwrite,
        ],
        [
            sys.executable,
            "scripts/transition_forecasting/data/validate_transition_run.py",
            "--dataset-1d",
            str(dataset_1d),
            "--dataset-3d",
            str(dataset_3d),
            "--report",
            str(validation / "data_pipeline_audit.json"),
        ],
        [
            sys.executable,
            "scripts/transition_forecasting/data/freeze_transition_checksums.py",
            "--dataset-1d",
            str(dataset_1d),
            "--dataset-3d",
            str(dataset_3d),
            "--report",
            str(validation / "data_pipeline_checksums.json"),
        ],
    ]

    for command in commands:
        run_command(command)

    print(f"Transition data ready: {root}")
    return 0
