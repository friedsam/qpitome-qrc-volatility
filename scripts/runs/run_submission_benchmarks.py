#!/usr/bin/env python3
"""Reproduce and validate the final Phase 3 QRC benchmark families.

This runner is intentionally separate from ``run_submission.py`` on the development
branch so it can be selectively ported into the clean integration branch. It uses
literal benchmark run IDs, never searches for the newest result directory, records
all commands, and writes SHA-256 inventories for every produced artifact.
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
DEFAULT_RUN_RESULTS_ROOT = REPO_ROOT / "results" / "runs"
DEFAULT_FOLD_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "global_transition_dataset_1d"
    / "purged_walk_forward_folds"
)
DEFAULT_MNIST_RAW_DIR = REPO_ROOT / "data" / "raw" / "mnist"

BENCHMARK_ROOTS: dict[str, Path] = {
    "mnist-palindrome": (
        REPO_ROOT
        / "results"
        / "transition_forecasting"
        / "qrc"
        / "mnist_palindrome_benchmark"
    ),
    "palindrome-noise": (
        REPO_ROOT
        / "results"
        / "transition_forecasting"
        / "qrc"
        / "palindrome_noise_assay"
    ),
    "palindrome-scaling": (
        REPO_ROOT
        / "results"
        / "transition_forecasting"
        / "qrc"
        / "palindrome_scaling_assay"
    ),
    "palindrome-shots": (
        REPO_ROOT
        / "results"
        / "transition_forecasting"
        / "qrc"
        / "palindrome_shot_assay"
    ),
}

REQUIRED_RELATIVE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "mnist-palindrome": (
        "params.json",
        "summary.json",
        "mnist_palindrome_features.npz",
        "shard_manifest.csv",
        "model_comparison.csv",
        "mnist_predictions.csv",
        "per_class_metrics.csv",
    ),
    "palindrome-noise": (
        "params.json",
        "summary.json",
        "noise_metrics.csv",
        "predictions.csv.gz",
        "retained_samples.csv",
        "frozen_readout.npz",
        "plots/forecast_metric_change_vs_ideal.png",
        "plots/feature_distortion_by_noise.png",
        "plots/feature_correlation_by_noise.png",
    ),
    "palindrome-scaling": (
        "params.json",
        "summary.json",
        "scaling_metrics.csv",
        "resource_scaling.csv",
        "predictions.csv.gz",
        "retained_samples.csv",
        "plots/validation_qlike_vs_atoms.png",
        "plots/runtime_scaling.png",
        "plots/statevector_memory_scaling.png",
    ),
    "palindrome-shots": (
        "params.json",
        "summary.json",
        "shot_metrics.csv",
        "shot_summary.csv",
        "direction_metrics.csv",
        "predictions.csv.gz",
        "retained_samples.csv",
        "exact_reference.npz",
        "plots/warning_gap_preservation_vs_shots.png",
        "plots/correction_correlation_vs_shots.png",
        "plots/transition_control_gap_vs_shots.png",
    ),
}

WORKFLOW_ORDER = (
    "mnist-palindrome",
    "palindrome-noise",
    "palindrome-scaling",
    "palindrome-shots",
)


@dataclass(frozen=True)
class BenchmarkIds:
    mnist: str
    noise: str
    scaling: str
    shots: str

    def for_workflow(self, workflow: str) -> str:
        return {
            "mnist-palindrome": self.mnist,
            "palindrome-noise": self.noise,
            "palindrome-scaling": self.scaling,
            "palindrome-shots": self.shots,
        }[workflow]


@dataclass
class CommandRecord:
    workflow: str
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
        "cwd": str(REPO_ROOT),
        "environment": {
            key: os.environ[key]
            for key in ("CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "QBRAID_ENVIRONMENT")
            if key in os.environ
        },
    }


def result_dir(workflow: str, run_id: str) -> Path:
    return BENCHMARK_ROOTS[workflow] / run_id


def required_output_inventory(
    workflow: str,
    run_id: str,
) -> dict[str, dict[str, object]]:
    root = result_dir(workflow, run_id)
    inventory: dict[str, dict[str, object]] = {}
    for relative in REQUIRED_RELATIVE_OUTPUTS[workflow]:
        path = root / relative
        item: dict[str, object] = {"exists": path.is_file()}
        if path.is_file():
            item.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
        inventory[relative] = item
    return inventory


def recursive_inventory(root: Path) -> dict[str, dict[str, object]]:
    if not root.is_dir():
        return {}
    output: dict[str, dict[str, object]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        output[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return output


def missing_required_outputs(workflow: str, run_id: str) -> list[str]:
    return [
        name
        for name, item in required_output_inventory(workflow, run_id).items()
        if not item["exists"]
    ]


def _python(script: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, script, *arguments)


def mnist_commands(
    *,
    profile: str,
    run_id: str,
    raw_dir: Path,
    shard_count: int,
    resume: bool,
) -> tuple[tuple[str, ...], ...]:
    if shard_count < 1:
        raise ValueError("mnist shard_count must be positive")
    if profile == "smoke":
        train_size, test_size, batch_size = 100, 20, 20
    elif profile == "primary":
        train_size, test_size, batch_size = 2000, 1000, 32
    else:
        raise ValueError(f"unsupported profile: {profile}")

    common = (
        "--train-size",
        str(train_size),
        "--test-size",
        str(test_size),
        "--batch-size",
        str(batch_size),
    )
    commands: list[tuple[str, ...]] = [
        _python(
            "scripts/transition_forecasting/data/acquire_mnist.py",
            "--destination",
            str(raw_dir),
            "--source-mode",
            "auto",
        )
    ]
    shard_dirs: list[str] = []
    for index in range(shard_count):
        shard_id = f"{run_id}_shard_{index:03d}_of_{shard_count:03d}"
        shard_dir = BENCHMARK_ROOTS["mnist-palindrome"] / shard_id
        shard_dirs.append(str(shard_dir))
        args = [
            "scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py",
            "shard",
            "--raw-dir",
            str(raw_dir),
            "--run-id",
            shard_id,
            "--shard-index",
            str(index),
            "--shard-count",
            str(shard_count),
            *common,
        ]
        if resume:
            args.append("--resume")
        commands.append(_python(*args))
    commands.append(
        _python(
            "scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py",
            "merge",
            "--run-id",
            run_id,
            "--shard-dirs",
            *shard_dirs,
            *common,
        )
    )
    return tuple(commands)


def workflow_commands(
    workflow: str,
    *,
    profile: str,
    ids: BenchmarkIds,
    fold_dir: Path,
    mnist_raw_dir: Path,
    mnist_shards: int,
    resume: bool,
) -> tuple[tuple[str, ...], ...]:
    if workflow == "mnist-palindrome":
        return mnist_commands(
            profile=profile,
            run_id=ids.mnist,
            raw_dir=mnist_raw_dir,
            shard_count=mnist_shards,
            resume=resume,
        )
    if workflow == "palindrome-noise":
        args = [
            "scripts/transition_forecasting/qrc/run_palindrome_noise_assay.py",
            "--fold-dir",
            str(fold_dir),
            "--run-id",
            ids.noise,
            "--fold",
            "5",
            "--lead",
            "5",
            "--max-per-class",
            "12",
        ]
        if profile == "smoke":
            args += ["--scenario-names", "depolarizing_p_0.01"]
        return (_python(*args),)
    if workflow == "palindrome-scaling":
        exact = (
            ["5", "6", "7"]
            if profile == "smoke"
            else [str(value) for value in range(5, 13)]
        )
        return (
            _python(
                "scripts/transition_forecasting/qrc/run_palindrome_scaling_assay.py",
                "--fold-dir",
                str(fold_dir),
                "--run-id",
                ids.scaling,
                "--fold",
                "5",
                "--lead",
                "5",
                "--max-per-class",
                "12",
                "--exact-atoms",
                *exact,
                "--resource-min-atoms",
                "5",
                "--resource-max-atoms",
                "20",
            ),
        )
    if workflow == "palindrome-shots":
        if profile == "smoke":
            shot_counts = ["100", "1000"]
            seeds = ["20260726"]
        else:
            shot_counts = [
                "100",
                "250",
                "500",
                "1000",
                "2000",
                "5000",
                "10000",
                "20000",
            ]
            seeds = [str(value) for value in range(20260717, 20260727)]
        return (
            _python(
                "scripts/transition_forecasting/qrc/run_palindrome_shot_assay.py",
                "--fold-dir",
                str(fold_dir),
                "--run-id",
                ids.shots,
                "--fold",
                "5",
                "--lead",
                "5",
                "--max-per-class",
                "12",
                "--shot-counts",
                *shot_counts,
                "--measurement-seeds",
                *seeds,
            ),
        )
    raise ValueError(f"unsupported workflow: {workflow}")


def selected_workflows(name: str) -> tuple[str, ...]:
    if name in {"all", "validate-existing"}:
        return WORKFLOW_ORDER
    if name in WORKFLOW_ORDER:
        return (name,)
    raise ValueError(f"unsupported workflow selection: {name}")


def run_command(
    workflow: str,
    argv: Sequence[str],
    run_dir: Path,
    index: int,
) -> CommandRecord:
    script_name = Path(argv[1]).stem if len(argv) > 1 else Path(argv[0]).stem
    log_path = run_dir / f"{index:02d}_{workflow}_{script_name}.log"
    started = utc_now()
    clock = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(value) for value in argv) + "\n\n")
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
        workflow=workflow,
        argv=list(argv),
        started_at_utc=started,
        finished_at_utc=utc_now(),
        duration_seconds=round(time.monotonic() - clock, 6),
        returncode=int(completed.returncode),
        log_path=str(log_path.relative_to(REPO_ROOT)),
    )


def execute(args: argparse.Namespace) -> tuple[int, Path]:
    run_dir = args.results_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "benchmark_manifest.json"
    ids = BenchmarkIds(
        mnist=args.mnist_run_id,
        noise=args.noise_run_id,
        scaling=args.scaling_run_id,
        shots=args.shots_run_id,
    )
    workflows = selected_workflows(args.workflow)
    validate_only = args.workflow == "validate-existing" or args.validate_only
    records: list[CommandRecord] = []
    status = "running"
    failure: dict[str, object] | None = None

    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "workflow_selection": args.workflow,
        "profile": args.profile,
        "validate_only": validate_only,
        "benchmark_run_ids": asdict(ids),
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": status,
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "working_tree_status": git_value("status", "--short") or "clean",
        },
        "environment": environment_snapshot(),
        "commands": [],
        "required_outputs": {},
        "artifact_inventories": {},
        "failure": None,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    command_index = 0
    for workflow in workflows:
        run_id = ids.for_workflow(workflow)
        if not validate_only:
            if args.reuse_existing and not missing_required_outputs(workflow, run_id):
                continue
            for command in workflow_commands(
                workflow,
                profile=args.profile,
                ids=ids,
                fold_dir=args.fold_dir,
                mnist_raw_dir=args.mnist_raw_dir,
                mnist_shards=args.mnist_shards,
                resume=args.resume,
            ):
                command_index += 1
                record = run_command(workflow, command, run_dir, command_index)
                records.append(record)
                if record.returncode != 0:
                    status = "failed"
                    failure = {
                        "reason": "command_failed",
                        "workflow": workflow,
                        "argv": record.argv,
                        "returncode": record.returncode,
                        "log_path": record.log_path,
                    }
                    break
        if status == "failed":
            break
        missing = missing_required_outputs(workflow, run_id)
        if missing:
            status = "failed"
            failure = {
                "reason": "missing_required_outputs",
                "workflow": workflow,
                "result_dir": str(
                    result_dir(workflow, run_id).relative_to(REPO_ROOT)
                ),
                "paths": missing,
            }
            break

    required: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    for workflow in workflows:
        run_id = ids.for_workflow(workflow)
        required[workflow] = required_output_inventory(workflow, run_id)
        artifacts[workflow] = {
            "result_dir": str(result_dir(workflow, run_id).relative_to(REPO_ROOT)),
            "files": recursive_inventory(result_dir(workflow, run_id)),
        }
    (run_dir / "benchmark_artifact_inventory.json").write_text(
        json.dumps(artifacts, indent=2) + "\n",
        encoding="utf-8",
    )

    if status != "failed":
        status = "succeeded"
    manifest.update(
        {
            "status": status,
            "finished_at_utc": utc_now(),
            "commands": [asdict(record) for record in records],
            "required_outputs": required,
            "artifact_inventories": {
                name: {
                    "result_dir": value["result_dir"],
                    "file_count": len(value["files"]),
                }
                for name, value in artifacts.items()
            },
            "failure": failure,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Benchmark workflow: {args.workflow}")
    print(f"Profile: {args.profile}")
    print(f"Status: {status}")
    print(f"Run directory: {run_dir.relative_to(REPO_ROOT)}")
    if failure:
        print(json.dumps(failure, indent=2))
    return (0 if status == "succeeded" else 1), run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or validate the frozen Phase 3 "
            "MNIST/noise/scaling/shot benchmarks."
        )
    )
    parser.add_argument(
        "workflow",
        choices=(*WORKFLOW_ORDER, "all", "validate-existing"),
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "primary"),
        default="primary",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Orchestrator manifest run ID.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RUN_RESULTS_ROOT,
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument(
        "--mnist-raw-dir",
        type=Path,
        default=DEFAULT_MNIST_RAW_DIR,
    )
    parser.add_argument("--mnist-shards", type=int, default=None)
    parser.add_argument("--mnist-run-id", default=None)
    parser.add_argument("--noise-run-id", default=None)
    parser.add_argument("--scaling-run-id", default=None)
    parser.add_argument("--shots-run-id", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume MNIST feature shards.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Skip a workflow only when its explicit result directory "
            "passes validation."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Do not execute commands; validate the explicit result "
            "directories only."
        ),
    )
    args = parser.parse_args(argv)
    suffix = "smoke_001" if args.profile == "smoke" else "primary_001"
    args.mnist_shards = args.mnist_shards or (
        1 if args.profile == "smoke" else 8
    )
    args.mnist_run_id = args.mnist_run_id or f"mnist_palindrome_{suffix}"
    args.noise_run_id = args.noise_run_id or f"palindrome_noise_{suffix}"
    args.scaling_run_id = (
        args.scaling_run_id or f"palindrome_scaling_{suffix}"
    )
    args.shots_run_id = args.shots_run_id or f"palindrome_shots_{suffix}"
    for attr in ("results_root", "fold_dir", "mnist_raw_dir"):
        path = getattr(args, attr)
        if not path.is_absolute():
            setattr(args, attr, REPO_ROOT / path)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))[0]


if __name__ == "__main__":
    raise SystemExit(main())
