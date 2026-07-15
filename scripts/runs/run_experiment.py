#!/usr/bin/env python3
"""Launch one scientific experiment into an immutable parameterized run directory.

Output contract:

    scripts/<area>/<script_name>.py
    -> results/<area>/<script_name>/<run-id>/

The launcher writes parameters and provenance before invoking the experiment and
passes the run directory through the experiment's ``--out-dir`` argument.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_script(value: str) -> Path:
    script = Path(value)
    if not script.is_absolute():
        script = REPO_ROOT / script
    script = script.resolve()

    scripts_root = (REPO_ROOT / "scripts").resolve()
    try:
        script.relative_to(scripts_root)
    except ValueError as exc:
        raise ValueError(f"Experiment script must be inside {scripts_root}: {script}") from exc

    if script.suffix != ".py" or not script.is_file():
        raise FileNotFoundError(f"Experiment script not found: {script}")
    return script


def results_root_for_script(script: Path) -> Path:
    relative = script.relative_to(REPO_ROOT / "scripts").with_suffix("")
    return REPO_ROOT / "results" / relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="Repository experiment script path")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "experiment_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the experiment; prefix with --",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = resolve_script(args.script)
    run_id = args.run_id or default_run_id()
    run_root = results_root_for_script(script)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    forwarded = list(args.experiment_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if "--out-dir" in forwarded:
        raise ValueError(
            "Do not pass --out-dir through run_experiment.py; the launcher owns the run directory."
        )

    command = [
        sys.executable,
        str(script.relative_to(REPO_ROOT)),
        *forwarded,
        "--out-dir",
        str(run_dir.relative_to(REPO_ROOT)),
    ]

    params = {
        "schema_version": 1,
        "run_id": run_id,
        "script": str(script.relative_to(REPO_ROOT)),
        "results_directory": str(run_dir.relative_to(REPO_ROOT)),
        "experiment_arguments": forwarded,
        "resolved_command": command,
        "created_at_utc": utc_now(),
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "working_tree_status": git_value("status", "--short") or "clean",
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cwd": str(REPO_ROOT),
            "variables": {
                key: os.environ[key]
                for key in ("CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "QBRAID_ENVIRONMENT")
                if key in os.environ
            },
        },
    }
    params_path = run_dir / "params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    status = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "finished_at_utc": utc_now(),
        "params": "params.json",
        "log": "run.log",
    }
    (run_dir / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )

    print(f"Run directory: {run_dir.relative_to(REPO_ROOT)}")
    print(f"Status: {status['status']}")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
