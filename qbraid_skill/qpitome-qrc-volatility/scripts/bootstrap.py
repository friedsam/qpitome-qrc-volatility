#!/usr/bin/env python3
"""Create the local environment and verify the qBraid skill contract.

This helper intentionally performs repository setup and focused contract
validation. Scientific execution remains owned by the established submission,
Case151, and Phase-3 benchmark runners.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VENV = REPO_ROOT / ".venv"
FOCUSED_TESTS = (
    "tests/qbraid_skill/test_skill_contract.py",
    "tests/transition_forecasting/data/test_acquisition.py",
    "tests/transition_forecasting/modeling/test_stage_d_candidate_pool.py",
    "tests/transition_forecasting/modeling/test_chronological_control_matching.py",
    "tests/transition_forecasting/modeling/test_chronological_rematched_dataset.py",
    "tests/transition_forecasting/modeling/test_chronological_splits.py",
    "tests/transition_forecasting/data/test_fold_datasets.py",
    "tests/runs/test_run_submission.py",
    "tests/runs/test_run_submission_classical.py",
    "tests/runs/test_run_submission_benchmarks.py",
    "tests/transition_forecasting/modeling/classical_benchmarks/test_common.py",
    "tests/transition_forecasting/modeling/classical_benchmarks/test_esn.py",
    "tests/transition_forecasting/modeling/classical_benchmarks/test_garch_mechanics.py",
    "tests/transition_forecasting/modeling/classical_benchmarks/test_spec.py",
    "tests/transition_forecasting/modeling/classical_benchmarks/test_validation.py",
    "tests/transition_forecasting/qrc/test_case151_reproduction_contract.py",
    "tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py",
    "tests/transition_forecasting/qrc/test_palindrome_rydberg_noise.py",
    "tests/transition_forecasting/qrc/test_palindrome_scaling_assay.py",
    "tests/transition_forecasting/qrc/test_palindrome_shot_assay.py",
)
REQUIRED_REPOSITORY_PATHS = (
    "pyproject.toml",
    "qbraid_skill/qpitome-qrc-volatility/SKILL.md",
    "qbraid_skill/qpitome-qrc-volatility/references/repository-map.md",
    "qbraid_skill/qpitome-qrc-volatility/references/run-contract.md",
    "qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py",
    "qbraid_skill/qpitome-qrc-volatility/scripts/preflight_classical.py",
    "scripts/runs/run_submission.py",
    "scripts/runs/run_submission_layout.py",
    "scripts/runs/run_submission_stage_layout.py",
    "scripts/runs/run_submission_benchmarks.py",
    "scripts/reproduction/run_case151_simulation.py",
    "config/transition_forecasting/classical_benchmarks/frozen_submission.json",
    "config/case151/expected_metrics.json",
    "config/case151/agent_run_spec.json",
    "scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/validate_classical_run.py",
    "scripts/transition_forecasting/data/acquire_mnist.py",
    "scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py",
    "scripts/transition_forecasting/qrc/run_palindrome_noise_assay.py",
    "scripts/transition_forecasting/qrc/run_palindrome_scaling_assay.py",
    "scripts/transition_forecasting/qrc/run_palindrome_shot_assay.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=DEFAULT_VENV,
        help="Virtual-environment directory (default: repository .venv)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the deterministic command plan without executing it",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Keep stdout machine-readable and print the plan or summary as JSON",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Install and preflight without running the focused test suite",
    )
    return parser.parse_args(argv)


def repository_failures() -> list[str]:
    failures: list[str] = []
    for relative in REQUIRED_REPOSITORY_PATHS:
        if not (REPO_ROOT / relative).is_file():
            failures.append(f"Missing required repository file: {relative}")
    return failures


def resolve_venv_dir(venv_dir: Path) -> Path:
    resolved = venv_dir.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def build_command_plan(venv_dir: Path, *, skip_tests: bool) -> list[list[str]]:
    resolved_venv = resolve_venv_dir(venv_dir)
    python_path = venv_python(resolved_venv)
    commands: list[list[str]] = []
    if not python_path.is_file():
        commands.append([sys.executable, "-m", "venv", str(resolved_venv)])
    commands.extend(
        [
            [str(python_path), "-m", "pip", "install", "-e", ".[test]"],
            [
                str(python_path),
                "qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py",
                "--json",
            ],
            [
                str(python_path),
                "qbraid_skill/qpitome-qrc-volatility/scripts/preflight_classical.py",
                "--json",
            ],
        ]
    )
    if not skip_tests:
        commands.append(
            [str(python_path), "-m", "pytest", "-q", *FOCUSED_TESTS]
        )
    return commands


def printable_command(argv: Sequence[str]) -> str:
    return shlex.join(str(value) for value in argv)


def run_command(argv: Sequence[str], *, machine_readable: bool) -> None:
    log_stream = sys.stderr if machine_readable else sys.stdout
    print(f"+ {printable_command(argv)}", file=log_stream, flush=True)
    environment = os.environ.copy()
    environment.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    completed = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=machine_readable,
        text=machine_readable,
    )
    if machine_readable:
        if completed.stdout:
            print(completed.stdout.rstrip(), file=sys.stderr)
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failures = repository_failures()
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    resolved_venv = resolve_venv_dir(args.venv_dir)
    plan = build_command_plan(resolved_venv, skip_tests=args.skip_tests)
    if args.plan:
        payload = {
            "repository_root": str(REPO_ROOT),
            "venv_dir": str(resolved_venv),
            "commands": plan,
            "skip_tests": bool(args.skip_tests),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for command in plan:
                print(printable_command(command))
        return 0
    completed_commands: list[list[str]] = []
    try:
        for command in plan:
            run_command(command, machine_readable=args.json)
            completed_commands.append(command)
    except subprocess.CalledProcessError as exc:
        print(
            f"ERROR: bootstrap command failed with exit code {exc.returncode}: "
            f"{printable_command(exc.cmd)}",
            file=sys.stderr,
        )
        return int(exc.returncode or 1)
    summary = {
        "status": "succeeded",
        "repository_root": str(REPO_ROOT),
        "venv_python": str(venv_python(resolved_venv)),
        "commands_completed": completed_commands,
        "focused_tests_run": not args.skip_tests,
        "focused_test_count": len(FOCUSED_TESTS) if not args.skip_tests else 0,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("qBraid skill bootstrap succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
