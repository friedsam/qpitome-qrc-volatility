#!/usr/bin/env python3
"""Reproducible submission workflow entry point.

The runner provides one stable interface for human, CI, and qBraid Agent reruns.
Each workflow writes a single aggregate package under ``results/runs/<run-id>``.
Scientific outputs are grouped by topic below ``files/`` and source code is never
copied into a result directory.
"""
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
TRANSITION_N_FOLDS = 8
FROZEN_CLASSICAL_SPEC = REPO_ROOT / "config" / "transition_forecasting" / "classical_benchmarks" / "frozen_submission.json"
CLASSICAL_MODELS = (
    "persistence", "har", "sequence_ridge", "garch_1_1_t",
    "esn_direct_tuned", "esn_shuffled_tuned",
)

DATA_COMMANDS = (
    (sys.executable, "scripts/data/download_market_data.py"),
    (sys.executable, "scripts/data/download_external_datasets.py"),
    (sys.executable, "scripts/data/inventory_raw_datasets.py"),
    (sys.executable, "scripts/data/build_volatility_dataset.py"),
    (sys.executable, "scripts/data/validate_volatility_dataset.py"),
    (sys.executable, "scripts/data/build_monthly_market_features.py"),
)
VALIDATE_DATA_COMMANDS = (
    (sys.executable, "scripts/data/inventory_raw_datasets.py"),
    (sys.executable, "scripts/data/validate_volatility_dataset.py"),
)
REQUIRED_DATA_OUTPUTS = tuple(
    REPO_ROOT / relative for relative in (
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
            ("git", *args),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def environment_snapshot() -> dict[str, object]:
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


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def output_inventory(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for path in paths:
        item: dict[str, object] = {"exists": path.is_file()}
        if path.is_file():
            item.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
        inventory[display_path(path)] = item
    return inventory


def transition_paths(run_dir: Path) -> dict[str, Path]:
    root = run_dir / "files" / "transition_forecasting"
    raw = root / "raw" / "global_stock_indices_historical_data"
    processed = root / "processed"
    dataset = processed / "global_transition_dataset_1d"
    validation = root / "validation"
    classical = root / "modeling" / "classical_baselines"
    return {
        "root": root,
        "raw": raw,
        "dataset_1d": dataset,
        "folds_1d": dataset / "purged_walk_forward_folds",
        "validation": validation / "data_pipeline_audit.json",
        "checksums": validation / "data_pipeline_checksums.json",
        "classical_root": classical,
        "linear_results_root": classical / "linear" / "run",
        "garch_results_root": classical / "garch" / "run",
        "esn_results_root": classical / "esn" / "run",
        "canonical_results_root": classical / "canonical" / "run",
        "classical_validation": validation / "classical_baseline_audit.json",
    }


def transition_commands(
    run_dir: Path,
    *,
    source_mode: str,
    force: bool,
) -> tuple[tuple[str, ...], ...]:
    paths = transition_paths(run_dir)
    overwrite = ("--force",) if force else ()
    return (
        (
            sys.executable,
            "scripts/transition_forecasting/data/acquire_global_index_data.py",
            "--destination",
            str(paths["raw"]),
            "--source-mode",
            source_mode,
            *overwrite,
        ),
        (
            sys.executable,
            "scripts/transition_forecasting/data/build_transition_datasets.py",
            "--raw-root",
            str(paths["raw"]),
            "--output-1d",
            str(paths["dataset_1d"]),
            *overwrite,
        ),
        (
            sys.executable,
            "scripts/transition_forecasting/data/build_transition_folds.py",
            "--dataset-1d",
            str(paths["dataset_1d"]),
            "--n-folds",
            str(TRANSITION_N_FOLDS),
            *overwrite,
        ),
        (
            sys.executable,
            "scripts/transition_forecasting/data/validate_transition_run.py",
            "--dataset-1d",
            str(paths["dataset_1d"]),
            "--report",
            str(paths["validation"]),
        ),
        (
            sys.executable,
            "scripts/transition_forecasting/data/freeze_transition_checksums.py",
            "--dataset-1d",
            str(paths["dataset_1d"]),
            "--report",
            str(paths["checksums"]),
        ),
    )


def classical_commands(run_dir: Path) -> tuple[tuple[str, ...], ...]:
    paths = transition_paths(run_dir)
    run_id = run_dir.name
    spec = str(FROZEN_CLASSICAL_SPEC)
    linear_run = paths["linear_results_root"] / run_id
    garch_run = paths["garch_results_root"] / run_id
    esn_run = paths["esn_results_root"] / run_id
    return (
        (
            sys.executable,
            "scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py",
            "--dataset-root",
            str(paths["dataset_1d"]),
            "--results-root",
            str(paths["linear_results_root"]),
            "--run-id",
            run_id,
            "--spec",
            spec,
        ),
        (
            sys.executable,
            "scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py",
            "--dataset-root",
            str(paths["dataset_1d"]),
            "--results-root",
            str(paths["garch_results_root"]),
            "--run-id",
            run_id,
            "--spec",
            spec,
        ),
        (
            sys.executable,
            "scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py",
            "--dataset-root",
            str(paths["dataset_1d"]),
            "--results-root",
            str(paths["esn_results_root"]),
            "--run-id",
            run_id,
            "--spec",
            spec,
        ),
        (
            sys.executable,
            "scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py",
            "--linear-run",
            str(linear_run),
            "--garch-run",
            str(garch_run),
            "--esn-run",
            str(esn_run),
            "--results-root",
            str(paths["canonical_results_root"]),
            "--run-id",
            run_id,
        ),
        (
            sys.executable,
            "scripts/transition_forecasting/modeling/classical_benchmarks/validate_classical_run.py",
            "--classical-root",
            str(paths["classical_root"]),
            "--run-id",
            run_id,
            "--report",
            str(paths["classical_validation"]),
        ),
    )


def command_plan(
    workflow: str,
    run_dir: Path,
    *,
    transition_source_mode: str,
    force: bool,
) -> tuple[tuple[str, ...], ...]:
    if workflow == "data":
        return DATA_COMMANDS
    if workflow == "validate-data":
        return VALIDATE_DATA_COMMANDS
    if workflow == "transition-data":
        return transition_commands(
            run_dir,
            source_mode=transition_source_mode,
            force=force,
        )
    if workflow == "financial-classical":
        return transition_commands(
            run_dir,
            source_mode=transition_source_mode,
            force=force,
        ) + classical_commands(run_dir)
    raise ValueError(f"Unsupported workflow: {workflow}")


def transition_required_outputs(run_dir: Path) -> tuple[Path, ...]:
    paths = transition_paths(run_dir)
    dataset_files = (
        "cleaned_ohlc.csv.gz",
        "daily_volatility.csv.gz",
        "transition_catalogue.csv",
        "sample_manifest.csv",
        "sequence_tensors.npz",
        "row_corrections.csv",
        "manifest.json",
        "control_candidate_manifest.csv",
        "control_candidate_tensors.npz",
        "candidate_pool_summary.json",
    )
    fold_files = (
        "rematched_rolling_manifest.csv",
        "rematched_rolling_tensors.npz",
        "control_match_audit.csv",
        "summary.json",
    )
    outputs: list[Path] = [
        paths["raw"] / "all_indices_data.csv",
        paths["raw"] / "raw_acquisition_manifest.json",
        paths["validation"],
        paths["checksums"],
    ]
    outputs.extend(paths["dataset_1d"] / name for name in dataset_files)
    outputs.extend(paths["folds_1d"] / name for name in fold_files)
    return tuple(outputs)


def classical_required_outputs(run_dir: Path) -> tuple[Path, ...]:
    paths = transition_paths(run_dir)
    run_id = run_dir.name

    def files(root: Path, *names: str) -> list[Path]:
        return [root / run_id / name for name in names]

    outputs: list[Path] = []
    outputs.extend(
        files(
            paths["linear_results_root"],
            "params.json",
            "config.json",
            "dataset_manifest.json",
            "predictions.csv.gz",
            "submission_metrics.csv",
            "metrics_by_fold.csv",
            "metrics_by_horizon.csv",
            "runtime.json",
            "summary.json",
        )
    )
    outputs.extend(
        files(
            paths["garch_results_root"],
            "params.json",
            "config.json",
            "dataset_manifest.json",
            "predictions.csv.gz",
            "submission_metrics.csv",
            "metrics_by_fold.csv",
            "metrics_by_horizon.csv",
            "fit_diagnostics.csv.gz",
            "runtime.json",
            "summary.json",
        )
    )
    outputs.extend(
        files(
            paths["esn_results_root"],
            "params.json",
            "config.json",
            "dataset_manifest.json",
            "selected_spec.json",
            "predictions.csv.gz",
            "submission_metrics.csv",
            "metrics_by_fold.csv",
            "metrics_by_horizon.csv",
            "runtime.json",
            "summary.json",
        )
    )
    outputs.extend(
        files(
            paths["canonical_results_root"],
            "params.json",
            "common_predictions.csv.gz",
            "submission_metrics.csv",
            "metrics_by_fold.csv",
            "metrics_by_horizon.csv",
            "coverage.csv",
            "paired_deltas_vs_sequence_ridge.csv",
            "submission_table_selection.csv",
            "submission_table_confirmation.csv",
            "submission_table_development_all.csv",
            "runtime.json",
            "summary.json",
        )
    )
    outputs.append(paths["classical_validation"])
    return tuple(outputs)


def required_outputs_for_workflow(
    workflow: str,
    run_dir: Path,
) -> tuple[Path, ...]:
    if workflow in {"data", "validate-data"}:
        return REQUIRED_DATA_OUTPUTS
    if workflow == "transition-data":
        return transition_required_outputs(run_dir)
    if workflow == "financial-classical":
        return transition_required_outputs(run_dir) + classical_required_outputs(run_dir)
    raise ValueError(f"Unsupported workflow: {workflow}")


def run_command(
    argv: Sequence[str],
    run_dir: Path,
    index: int,
) -> CommandRecord:
    command_name = Path(argv[1]).stem if len(argv) > 1 else Path(argv[0]).stem
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{index:02d}_{command_name}.log"
    started = utc_now()
    start_clock = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(argv)}\n\n")
        log.flush()
        completed = subprocess.run(
            tuple(argv),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return CommandRecord(
        argv=list(argv),
        started_at_utc=started,
        finished_at_utc=utc_now(),
        duration_seconds=round(time.monotonic() - start_clock, 6),
        returncode=completed.returncode,
        log_path=display_path(log_path),
    )


def execute(
    workflow: str,
    results_root: Path,
    run_id: str | None = None,
    *,
    transition_source_mode: str = "auto",
    force: bool = False,
) -> tuple[int, Path]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    records: list[CommandRecord] = []
    status = "running"
    failure: dict[str, object] | None = None
    commands = command_plan(
        workflow,
        run_dir,
        transition_source_mode=transition_source_mode,
        force=force,
    )
    transition_workflow = workflow in {"transition-data", "financial-classical"}
    manifest: dict[str, object] = {
        "schema_version": 3,
        "run_id": run_id,
        "workflow": workflow,
        "status": status,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "run_directory": display_path(run_dir),
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "working_tree_status": git_value("status", "--short") or "clean",
        },
        "environment": environment_snapshot(),
        "parameters": {
            "transition_source_mode": transition_source_mode,
            "transition_n_folds": TRANSITION_N_FOLDS if transition_workflow else None,
            "transition_channels": ["log_volatility_level"] if transition_workflow else None,
            "classical_models": list(CLASSICAL_MODELS) if workflow == "financial-classical" else None,
            "frozen_classical_spec": display_path(FROZEN_CLASSICAL_SPEC) if workflow == "financial-classical" else None,
            "test_evaluated": False if transition_workflow else None,
            "force": force,
        },
        "commands": [],
        "required_outputs": {},
        "failure": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for index, argv in enumerate(commands, start=1):
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
    required_outputs = output_inventory(
        required_outputs_for_workflow(workflow, run_dir)
    )
    missing_outputs = [
        path for path, item in required_outputs.items() if not bool(item["exists"])
    ]
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
    print(f"Run directory: {display_path(run_dir)}")
    print(f"Manifest: {display_path(manifest_path)}")
    if failure:
        print(json.dumps(failure, indent=2))
    return (0 if status == "succeeded" else 1), run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible submission workflows and record provenance"
    )
    parser.add_argument(
        "workflow",
        choices=("data", "validate-data", "transition-data", "financial-classical"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--transition-source-mode",
        choices=("auto", "live", "fallback"),
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = (
        args.results_root
        if args.results_root.is_absolute()
        else REPO_ROOT / args.results_root
    )
    return execute(
        args.workflow,
        results_root,
        args.run_id,
        transition_source_mode=args.transition_source_mode,
        force=args.force,
    )[0]


if __name__ == "__main__":
    raise SystemExit(main())
