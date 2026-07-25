#!/usr/bin/env python3
"""Validate the qBraid execution environment and Stage-1 repository contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from transition_forecasting.data.acquisition import validate_source, verify_fallback_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
MIN_PYTHON = (3, 10)
DATASET = "guillemservera/global-stock-indices-historical-data"
FALLBACK_ROOT = (
    REPO_ROOT
    / "data/fallback/transition_forecasting/global_stock_indices_historical_data"
)
FALLBACK_MANIFEST = FALLBACK_ROOT / "fallback_manifest.json"

REQUIRED_PATHS = (
    "pyproject.toml",
    "environment.yml",
    "qbraid_skill/qpitome-qrc-volatility/SKILL.md",
    "qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py",
    "qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py",
    "scripts/runs/run_submission.py",
    "scripts/transition_forecasting/data/acquire_global_index_data.py",
    "scripts/transition_forecasting/data/build_transition_datasets.py",
    "scripts/transition_forecasting/data/build_transition_folds.py",
    "scripts/transition_forecasting/data/validate_transition_run.py",
    "scripts/transition_forecasting/data/freeze_transition_checksums.py",
    "config/transition_forecasting/contracts/global_index_ohlc_inventory.csv",
    "config/transition_forecasting/contracts/global_range_quality.csv",
)

REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "sklearn",
    "kaggle",
)


def command_output(argv: Sequence[str], *, timeout: int = 30) -> dict[str, object]:
    """Run a read-only command and return a JSON-safe result."""

    try:
        completed = subprocess.run(
            tuple(argv),
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_snapshot() -> dict[str, object]:
    commit = command_output(("git", "rev-parse", "HEAD"))
    branch = command_output(("git", "branch", "--show-current"))
    status = command_output(("git", "status", "--short"))
    root = command_output(("git", "rev-parse", "--show-toplevel"))
    return {
        "root": root["stdout"] if root["returncode"] == 0 else None,
        "commit": commit["stdout"] if commit["returncode"] == 0 else None,
        "branch": branch["stdout"] if branch["returncode"] == 0 else None,
        "working_tree_status": status["stdout"] if status["returncode"] == 0 else None,
        "clean": bool(status["returncode"] == 0 and not status["stdout"]),
    }


def module_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}


def credential_status() -> dict[str, object]:
    home = Path.home()
    candidates = (
        home / ".kaggle/access_token",
        home / ".kaggle/kaggle.json",
    )
    records = []
    for path in candidates:
        exists = path.is_file()
        mode = oct(path.stat().st_mode & 0o777) if exists else None
        records.append(
            {
                "path": str(path),
                "exists": exists,
                "mode": mode,
                "secure_mode": bool(exists and (path.stat().st_mode & 0o077) == 0),
            }
        )
    environment_configured = bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )
    return {
        "credential_files": records,
        "environment_configured": environment_configured,
        "configured": bool(
            environment_configured or any(bool(item["exists"]) for item in records)
        ),
        "secure": bool(
            environment_configured or any(bool(item["secure_mode"]) for item in records)
        ),
    }


def environment_executable(name: str) -> str | None:
    """Prefer an executable installed beside the active Python interpreter."""

    candidate = Path(sys.executable).with_name(name)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(name)


def fallback_status() -> dict[str, object]:
    if not FALLBACK_MANIFEST.is_file():
        return {
            "manifest": str(FALLBACK_MANIFEST),
            "available": False,
            "verified": False,
            "verification": None,
            "error": "fallback manifest is absent",
        }
    try:
        verification = verify_fallback_manifest(FALLBACK_ROOT)
        validation = validate_source(FALLBACK_ROOT)
    except Exception as exc:
        return {
            "manifest": str(FALLBACK_MANIFEST),
            "available": True,
            "verified": False,
            "verification": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "manifest": str(FALLBACK_MANIFEST),
        "available": True,
        "verified": True,
        "verification": verification,
        "validation": validation,
        "error": None,
    }


def build_report() -> dict[str, object]:
    required = {
        relative: (REPO_ROOT / relative).is_file() for relative in REQUIRED_PATHS
    }
    missing_required = [path for path, exists in required.items() if not exists]

    qbraid_path = shutil.which("qbraid")
    kaggle_path = environment_executable("kaggle")
    qbraid_version = (
        command_output((qbraid_path, "--version")) if qbraid_path else None
    )
    kaggle_version = (
        command_output((kaggle_path, "--version")) if kaggle_path else None
    )
    modules = module_status()
    credentials = credential_status()
    fallback = fallback_status()
    kaggle_probe = (
        command_output((kaggle_path, "datasets", "files", DATASET), timeout=60)
        if kaggle_path and modules.get("kaggle", False)
        else None
    )

    live_ready = bool(kaggle_probe and kaggle_probe.get("returncode") == 0)
    fallback_ready = bool(fallback["verified"])

    core_failures: list[str] = []
    if sys.version_info < MIN_PYTHON:
        core_failures.append(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required"
        )
    if missing_required:
        core_failures.append(
            "Missing required repository paths: " + ", ".join(missing_required)
        )
    missing_modules = [name for name, available in modules.items() if not available]
    if missing_modules:
        core_failures.append(
            "Missing installed Python modules: " + ", ".join(missing_modules)
        )

    warnings: list[str] = []
    if qbraid_path is None:
        warnings.append(
            "qBraid CLI was not found on PATH; this is expected outside qBraid Lab"
        )
    if not fallback["available"]:
        warnings.append(
            "The verified transition-data fallback is not present in this checkout"
        )
    elif not fallback_ready:
        warnings.append(
            "The transition-data fallback is present but failed verification: "
            + str(fallback["error"])
        )
    if credentials["configured"] and not credentials["secure"]:
        warnings.append(
            "A Kaggle credential file exists but does not have a restrictive permission mode"
        )
    if kaggle_probe and kaggle_probe.get("returncode") != 0:
        detail = str(kaggle_probe.get("stderr") or kaggle_probe.get("stdout") or "")
        warnings.append(
            "Anonymous Kaggle dataset access probe failed"
            + (f": {detail[:300]}" if detail else "")
        )
    if not live_ready and not fallback_ready:
        warnings.append(
            "No verified transition-data source is available: anonymous Kaggle access or the verified fallback is required"
        )

    recommended_mode = (
        "auto"
        if live_ready and fallback_ready
        else "fallback"
        if fallback_ready
        else "live"
        if live_ready
        else None
    )

    return {
        "schema_version": 2,
        "repository_root": str(REPO_ROOT),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "minimum_supported": f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            "supported": sys.version_info >= MIN_PYTHON,
            "repository_local_venv": Path(sys.executable).absolute().is_relative_to(
                (REPO_ROOT / ".venv").absolute()
            ),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "qbraid_environment": os.environ.get("QBRAID_ENVIRONMENT"),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "virtual_environment": os.environ.get("VIRTUAL_ENV"),
        },
        "git": git_snapshot(),
        "repository_contract": {
            "required_paths": required,
            "missing_required_paths": missing_required,
        },
        "python_modules": modules,
        "qbraid_cli": {
            "path": qbraid_path,
            "version": qbraid_version,
        },
        "kaggle": {
            "dataset": DATASET,
            "path": kaggle_path,
            "version": kaggle_version,
            "dataset_files_probe": kaggle_probe,
            "anonymous_access_ready": live_ready,
            **credentials,
        },
        "fallback": fallback,
        "data_source": {
            "live_ready": live_ready,
            "fallback_ready": fallback_ready,
            "ready": bool(live_ready or fallback_ready),
            "recommended_mode": recommended_mode,
        },
        "core_failures": core_failures,
        "warnings": warnings,
        "passed": not core_failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the complete machine-readable report",
    )
    parser.add_argument(
        "--strict-data-source",
        action="store_true",
        help="Fail unless anonymous Kaggle access or a verified fallback is available",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Repository: {report['repository_root']}")
        print(f"Python: {report['python']['version']}")
        print(f"Core contract: {'PASS' if report['passed'] else 'FAIL'}")
        print(
            "Data source: "
            + (
                str(report["data_source"]["recommended_mode"])
                if report["data_source"]["ready"]
                else "UNAVAILABLE"
            )
        )
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for failure in report["core_failures"]:
            print(f"ERROR: {failure}")

    if not report["passed"]:
        return 1
    if args.strict_data_source and not report["data_source"]["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
