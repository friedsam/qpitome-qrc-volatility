#!/usr/bin/env python3
"""Submission workflow entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "runs"

DATA_COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "scripts/data/download_market_data.py"),
    (sys.executable, "scripts/data/download_external_datasets.py"),
    (sys.executable, "scripts/data/inventory_raw_datasets.py"),
    (sys.executable, "scripts/data/build_volatility_dataset.py"),
    (sys.executable, "scripts/data/validate_volatility_dataset.py"),
    (sys.executable, "scripts/data/build_monthly_market_features.py"),
)

VALIDATE_DATA_COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "scripts/data/inventory_raw_datasets.py"),
    (sys.executable, "scripts/data/validate_volatility_dataset.py"),
)

TRANSITION_RAW_DATA_COMMANDS: tuple[tuple[str, ...], ...] = (
    (
        sys.executable,
        "scripts/transition_forecasting/quality/fetch_global_index_ohlc.py",
        "--source-mode",
        "auto",
    ),
)

TRANSITION_PROCESS_COMMANDS: tuple[tuple[str, ...], ...] = (
    (
        sys.executable,
        "scripts/transition_forecasting/build_processed_dataset.py",
        "--force",
    ),
)

REQUIRED_DATA_OUTPUTS: tuple[str, ...] = (
    "data/raw/yahoo_daily_history/historical_market_data.csv",
    "data/raw/yahoo_daily_history/historical_volatility_data.csv",
    "data/raw/external_dataset_manifest.json",
    "data/raw/raw_dataset_inventory.json",
    "data/processed/spy_vix_volatility/spy_vix_volatility.csv",
    "data/processed/monthly_market_features/paper_parity_1950_02_to_2017_12.parquet",
    "data/processed/monthly_market_features/extended_complete_market_months.parquet",
    "data/processed/monthly_market_features/features/preliminary_features.parquet",
    "data/processed/monthly_market_features/features/feature_catalog.csv",
    "data/processed/monthly_market_features/manifest.json",
)

REQUIRED_TRANSITION_RAW_DATA_OUTPUTS: tuple[str, ...] = (
    "data/raw/transition_forecasting/global_stock_indices_historical_data/all_indices_data.csv",
    "data/raw/transition_forecasting/global_stock_indices_historical_data/source_manifest.json",
    "data/raw/transition_forecasting/global_stock_indices_historical_data/raw_acquisition_manifest.json",
)

REQUIRED_TRANSITION_PROCESS_OUTPUTS: tuple[str, ...] = (
    "data/processed/transition_forecasting/global_transition_dataset/cleaned_ohlc.csv.gz",
    "data/processed/transition_forecasting/global_transition_dataset/daily_volatility.csv.gz",
    "data/processed/transition_forecasting/global_transition_dataset/transition_catalogue.csv",
    "data/processed/transition_forecasting/global_transition_dataset/sample_manifest.csv",
    "data/processed/transition_forecasting/global_transition_dataset/sequence_tensors.npz",
    "data/processed/transition_forecasting/global_transition_dataset/row_corrections.csv",
    "data/processed/transition_forecasting/global_transition_dataset/manifest.json",
)


@dataclass
class CommandRecord:
    argv: list[str]
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    returncode: int
    log_path: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args), cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def environment_snapshot() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(REPO_ROOT),
        "environment": {
            key: os.environ[key]
            for key in ("CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "QBRAID_ENVIRONMENT")
            if key in os.environ
        },
    }


def output_inventory(paths: Sequence[str]) -> dict[str, dict]:
    inventory: dict[str, dict] = {}
    for relative in paths:
        path = REPO_ROOT / relative
        item = {"exists": path.is_file()}
        if path.is_file():
            item.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
        inventory[relative] = item
    return inventory


def run_command(argv: Sequence[str], run_dir: Path, index: int) -> CommandRecord:
    command_name = Path(argv[1]).stem if len(argv) > 1 else Path(argv[0]).stem
    log_path = run_dir / f"{index:02d}_{command_name}.log"
    started = utc_now()
    start_clock = time.monotonic()
    actual_argv = list(argv)
    script_names = {Path(argument).name for argument in actual_argv}
    if "build_processed_dataset.py" in script_names:
        actual_argv.extend(["--log-dir", str(run_dir)])
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(actual_argv)}\n\n")
        log.flush()
        completed = subprocess.run(
            tuple(actual_argv),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return CommandRecord(
        argv=actual_argv,
        started_at_utc=started,
        finished_at_utc=utc_now(),
        duration_seconds=round(time.monotonic() - start_clock, 6),
        returncode=completed.returncode,
        log_path=str(log_path.relative_to(REPO_ROOT)),
    )


def command_plan(workflow: str) -> tuple[tuple[str, ...], ...]:
    if workflow == "data":
        return DATA_COMMANDS
    if workflow == "validate-data":
        return VALIDATE_DATA_COMMANDS
    if workflow == "transition-raw-data":
        return TRANSITION_RAW_DATA_COMMANDS
    if workflow == "transition-process":
        return TRANSITION_PROCESS_COMMANDS
    raise ValueError(f"Unsupported workflow: {workflow}")


def required_outputs_for_workflow(workflow: str) -> tuple[str, ...]:
    if workflow in {"data", "validate-data"}:
        return REQUIRED_DATA_OUTPUTS
    if workflow == "transition-raw-data":
        return REQUIRED_TRANSITION_RAW_DATA_OUTPUTS
    if workflow == "transition-process":
        return REQUIRED_TRANSITION_PROCESS_OUTPUTS
    raise ValueError(f"Unsupported workflow: {workflow}")


def execute(workflow: str, results_root: Path, run_id: str | None = None) -> tuple[int, Path]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    records: list[CommandRecord] = []
    status = "running"
    failure: dict | None = None
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "workflow": workflow,
        "status": status,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "working_tree_status": git_value("status", "--short") or "clean",
        },
        "environment": environment_snapshot(),
        "commands": [],
        "required_outputs": {},
        "failure": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for index, argv in enumerate(command_plan(workflow), start=1):
        record = run_command(argv, run_dir, index)
        records.append(record)
        if record.returncode != 0:
            status = "failed"
            failure = {
                "command_index": index,
                "argv": record.argv,
                "returncode": record.returncode,
                "log_path": record.log_path,
            }
            break

    required_outputs = output_inventory(required_outputs_for_workflow(workflow))
    missing_outputs = [path for path, item in required_outputs.items() if not item["exists"]]
    if status != "failed" and missing_outputs:
        status = "failed"
        failure = {"reason": "missing_required_outputs", "paths": missing_outputs}
    elif status != "failed":
        status = "succeeded"

    manifest.update(
        {
            "status": status,
            "finished_at_utc": utc_now(),
            "commands": [asdict(record) for record in records],
            "required_outputs": required_outputs,
            "failure": failure,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Workflow: {workflow}")
    print(f"Status: {status}")
    print(f"Run directory: {run_dir.relative_to(REPO_ROOT)}")
    print(f"Manifest: {manifest_path.relative_to(REPO_ROOT)}")
    if failure:
        print(json.dumps(failure, indent=2))
    return (0 if status == "succeeded" else 1), run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible submission workflows and record provenance")
    parser.add_argument(
        "workflow",
        choices=("data", "validate-data", "transition-raw-data", "transition-process"),
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = args.results_root
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root
    return execute(args.workflow, results_root, args.run_id)[0]


if __name__ == "__main__":
    raise SystemExit(main())
