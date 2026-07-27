#!/usr/bin/env python3
"""Run the judge-facing submission scope without relying on shell state.

This is an orchestration helper only. It delegates every scientific stage to the
existing canonical entry points and computes all paths from one literal run ID.
It never submits, queries, selects, or retrieves hardware jobs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "runs"


@dataclass
class CommandRecord:
    argv: list[str]
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    returncode: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("qbraid-submission-%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _current_case151_audit_is_verified(run_dir: Path, run_id: str) -> bool:
    audit_path = (
        run_dir
        / "files"
        / "qrc"
        / "simulation"
        / "run"
        / run_id
        / "case151_reproduction_audit.json"
    )
    if not audit_path.is_file():
        return False
    try:
        audit = read_json(audit_path)
    except Exception:
        return False
    return (
        audit.get("status") == "verified"
        and audit.get("verification_mode") == "current-pipeline"
        and audit.get("historical_reference_hashes_verified") is True
    )


def command_plan(
    *,
    scope: str,
    run_id: str,
    run_dir: Path,
    transition_source_mode: str,
    resume_existing: bool,
) -> tuple[tuple[str, ...], ...]:
    python = sys.executable
    commands: list[tuple[str, ...]] = []

    if not resume_existing:
        commands.append(
            (
                python,
                "scripts/runs/run_submission_stage_layout.py",
                "financial-classical",
                "--run-id",
                run_id,
                "--transition-source-mode",
                transition_source_mode,
            )
        )

    if not (
        resume_existing and _current_case151_audit_is_verified(run_dir, run_id)
    ):
        commands.append(
            (
                python,
                "scripts/reproduction/run_case151_simulation.py",
                "--fold-dir",
                str(
                    run_dir
                    / "files"
                    / "data"
                    / "processed"
                    / "global_transition_dataset_1d"
                    / "purged_walk_forward_folds"
                ),
                "--output-root",
                str(run_dir / "files" / "qrc" / "simulation" / "run"),
                "--run-id",
                run_id,
                "--verification-mode",
                "current-pipeline",
                "--archive-existing-failed",
            )
        )

    if scope == "full-smoke":
        commands.extend(
            (
                (
                    python,
                    "scripts/runs/run_submission_benchmarks.py",
                    "all",
                    "--profile",
                    "smoke",
                    "--run-id",
                    run_id,
                    "--run-dir",
                    str(run_dir),
                    "--resume",
                    "--reuse-existing",
                ),
                (
                    python,
                    "scripts/runs/run_submission_benchmarks.py",
                    "validate-existing",
                    "--profile",
                    "smoke",
                    "--run-id",
                    run_id,
                    "--run-dir",
                    str(run_dir),
                ),
            )
        )

    return tuple(commands)


def validate_classical(run_dir: Path) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing aggregate manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "succeeded":
        raise RuntimeError(f"aggregate workflow did not succeed: {manifest.get('status')}")
    if manifest.get("workflow") != "financial-classical":
        raise RuntimeError(
            f"unexpected aggregate workflow: {manifest.get('workflow')!r}"
        )
    if manifest.get("parameters", {}).get("test_evaluated") is not False:
        raise RuntimeError("aggregate manifest does not preserve test_evaluated=false")

    audit_path = (
        run_dir
        / "files"
        / "classical_baselines"
        / "validation"
        / "classical_baseline_audit.json"
    )
    audit = read_json(audit_path)
    if audit.get("passed") is not True:
        raise RuntimeError(f"classical audit failed: {audit_path}")
    if audit.get("test_evaluated") is not False:
        raise RuntimeError("classical audit does not preserve test_evaluated=false")


def validate_case151(run_dir: Path, run_id: str) -> None:
    qrc_dir = run_dir / "files" / "qrc" / "simulation" / "run" / run_id
    audit_path = qrc_dir / "case151_reproduction_audit.json"
    audit = read_json(audit_path)
    expected = {
        "status": "verified",
        "verification_mode": "current-pipeline",
        "feature_bank": "occupation_pair_raw",
        "feature_width": 63,
        "fold8_selected_alpha": 0.1,
        "fold8_selected_lambda": 0.25,
        "test_rows_used": 0,
        "qrc_head_fit_intercept": False,
        "historical_metric_oracle_applied": False,
        "historical_reference_hashes_verified": True,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise RuntimeError(
                f"Case151 audit mismatch for {key}: "
                f"observed={audit.get(key)!r}, expected={value!r}"
            )
    if not isinstance(audit.get("observed_metrics"), dict):
        raise RuntimeError("Case151 audit is missing current-pipeline metrics")
    if not isinstance(audit.get("metric_deltas_vs_historical_reference"), dict):
        raise RuntimeError("Case151 audit is missing historical metric deltas")
    for name in (
        "pooled_metrics.csv",
        "readout_selections.csv",
        "feature_diagnostics.csv",
        "summary.json",
    ):
        if not (qrc_dir / name).is_file():
            raise RuntimeError(f"missing Case151 output: {qrc_dir / name}")


def validate_smoke(run_dir: Path) -> None:
    root = run_dir / "files" / "quantum_studies"
    manifest_path = root / "benchmark_manifest.json"
    inventory_path = root / "benchmark_artifact_inventory.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "succeeded":
        raise RuntimeError(f"benchmark validation failed: {manifest_path}")
    if manifest.get("profile") != "smoke":
        raise RuntimeError("benchmark manifest does not record profile=smoke")
    if manifest.get("validate_only") is not True:
        raise RuntimeError("final benchmark pass is not validate-only")
    if manifest.get("failure") is not None:
        raise RuntimeError(f"benchmark manifest records failure: {manifest['failure']}")
    if not inventory_path.is_file():
        raise RuntimeError(f"missing benchmark inventory: {inventory_path}")

    expected = {
        "mnist-palindrome",
        "palindrome-noise",
        "palindrome-scaling",
        "palindrome-shots",
    }
    required = manifest.get("required_outputs", {})
    if set(required) != expected:
        raise RuntimeError(f"unexpected benchmark workflow set: {sorted(required)}")
    for workflow, files in required.items():
        missing = [
            name
            for name, record in files.items()
            if not bool(record.get("exists"))
        ]
        if missing:
            raise RuntimeError(f"{workflow} missing required outputs: {missing}")


def validate_no_source_files(run_dir: Path) -> None:
    source_files = sorted(run_dir.rglob("*.py"))
    if source_files:
        raise RuntimeError(
            "source files found beneath aggregate run: "
            + ", ".join(str(path) for path in source_files)
        )


def write_scope_manifest(
    *,
    path: Path,
    run_id: str,
    scope: str,
    status: str,
    started_at_utc: str,
    records: Sequence[CommandRecord],
    failure: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "scope": scope,
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now(),
        "commands": [asdict(record) for record in records],
        "hardware_actions_performed": False,
        "case151_verification_mode": "current-pipeline",
        "historical_case151_metric_oracle_relabelled": False,
        "failure": failure,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(argv: Sequence[str]) -> CommandRecord:
    started = utc_now()
    clock = time.monotonic()
    print("+ " + " ".join(str(value) for value in argv), flush=True)
    completed = subprocess.run(tuple(argv), cwd=REPO_ROOT, check=False)
    return CommandRecord(
        argv=[str(value) for value in argv],
        started_at_utc=started,
        finished_at_utc=utc_now(),
        duration_seconds=round(time.monotonic() - clock, 6),
        returncode=int(completed.returncode),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("core", "full-smoke"),
        default="full-smoke",
    )
    parser.add_argument(
        "--transition-source-mode",
        choices=("auto", "live", "fallback"),
        required=True,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume an accepted financial-classical run using --run-id.",
    )
    args = parser.parse_args(argv)
    if args.resume_existing and not args.run_id:
        parser.error("--resume-existing requires --run-id")
    if not args.results_root.is_absolute():
        args.results_root = REPO_ROOT / args.results_root
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or default_run_id()
    run_dir = args.results_root / run_id
    scope_manifest = run_dir / "agent_scope_manifest.json"
    started_at = utc_now()
    records: list[CommandRecord] = []
    failure: str | None = None

    if run_dir.exists() and not args.resume_existing:
        print(
            f"ERROR: run directory already exists; use a new run ID: {run_dir}",
            file=sys.stderr,
        )
        return 2
    if args.resume_existing and not run_dir.is_dir():
        print(f"ERROR: resume run directory is absent: {run_dir}", file=sys.stderr)
        return 2

    try:
        if args.resume_existing:
            validate_classical(run_dir)

        for command in command_plan(
            scope=args.scope,
            run_id=run_id,
            run_dir=run_dir,
            transition_source_mode=args.transition_source_mode,
            resume_existing=args.resume_existing,
        ):
            record = run_command(command)
            records.append(record)
            if record.returncode != 0:
                raise RuntimeError(
                    f"command failed with exit code {record.returncode}: "
                    + " ".join(record.argv)
                )

            if command[1].endswith("run_submission_stage_layout.py"):
                validate_classical(run_dir)
            elif command[1].endswith("run_case151_simulation.py"):
                validate_case151(run_dir, run_id)

        validate_classical(run_dir)
        validate_case151(run_dir, run_id)
        if args.scope == "full-smoke":
            validate_smoke(run_dir)
        validate_no_source_files(run_dir)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if run_dir.exists():
            write_scope_manifest(
                path=scope_manifest,
                run_id=run_id,
                scope=args.scope,
                status="failed",
                started_at_utc=started_at,
                records=records,
                failure=failure,
            )
        print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    write_scope_manifest(
        path=scope_manifest,
        run_id=run_id,
        scope=args.scope,
        status="succeeded",
        started_at_utc=started_at,
        records=records,
        failure=None,
    )
    print(f"Submission scope: {args.scope}")
    print("Status: succeeded")
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Scope manifest: {scope_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
