#!/usr/bin/env python3
"""Validate the frozen classical submission contract without running science."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "arch",
)
REQUIRED_PATHS = (
    "config/transition_forecasting/classical_benchmarks/frozen_submission.json",
    "src/transition_forecasting/modeling/classical_benchmarks/common.py",
    "src/transition_forecasting/modeling/classical_benchmarks/linear.py",
    "src/transition_forecasting/modeling/classical_benchmarks/garch.py",
    "src/transition_forecasting/modeling/classical_benchmarks/esn.py",
    "src/transition_forecasting/modeling/classical_benchmarks/esn_frozen.py",
    "src/transition_forecasting/modeling/classical_benchmarks/canonical.py",
    "src/transition_forecasting/modeling/classical_benchmarks/validation.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py",
    "scripts/transition_forecasting/modeling/classical_benchmarks/validate_classical_run.py",
)


def build_report() -> dict[str, object]:
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in REQUIRED_MODULES
    }
    paths = {
        relative: (REPO_ROOT / relative).is_file()
        for relative in REQUIRED_PATHS
    }
    failures = [
        f"missing Python module: {name}"
        for name, available in modules.items()
        if not available
    ]
    failures.extend(
        f"missing repository path: {path}"
        for path, exists in paths.items()
        if not exists
    )
    spec_path = (
        REPO_ROOT
        / "config"
        / "transition_forecasting"
        / "classical_benchmarks"
        / "frozen_submission.json"
    )
    spec: dict[str, object] | None = None
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid frozen specification: {type(exc).__name__}: {exc}")
    if spec is not None:
        if spec.get("reporting", {}).get("test_evaluated") is not False:
            failures.append("frozen specification does not keep test_evaluated=false")
        if spec.get("garch", {}).get("backend") != "arch":
            failures.append("frozen GARCH backend is not arch")
        if spec.get("selection_folds") != [4, 5, 6]:
            failures.append("frozen selection folds differ from 4-6")
        if spec.get("confirmation_folds") != [7, 8]:
            failures.append("frozen confirmation folds differ from 7-8")
    return {
        "schema_version": 1,
        "repository_root": str(REPO_ROOT),
        "python": sys.version,
        "modules": modules,
        "paths": paths,
        "frozen_spec": spec,
        "failures": failures,
        "passed": not failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Classical contract: {'PASS' if report['passed'] else 'FAIL'}")
        for failure in report["failures"]:
            print(f"ERROR: {failure}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
