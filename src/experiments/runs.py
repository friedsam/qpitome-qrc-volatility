"""Shared run-directory creation and provenance capture for experiment scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _git_value(*args: str) -> str | None:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {
        "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return record


def environment_snapshot() -> dict[str, Any]:
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


@dataclass
class RunContext:
    run_dir: Path
    workflow: str
    parameters: dict[str, Any]
    manifest_path: Path = field(init=False)
    manifest: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.manifest = {
            "schema_version": 1,
            "run_id": self.run_dir.name,
            "workflow": self.workflow,
            "status": "running",
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "repository": {
                "commit": _git_value("rev-parse", "HEAD"),
                "branch": _git_value("branch", "--show-current"),
                "working_tree_status": _git_value("status", "--short") or "clean",
            },
            "environment": environment_snapshot(),
            "parameters": _json_safe(self.parameters),
            "inputs": {},
            "outputs": {},
            "failure": None,
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record_input(self, name: str, path: Path) -> None:
        self.manifest["inputs"][name] = file_record(Path(path))
        self._write_manifest()

    def record_output(self, name: str, path: Path) -> None:
        self.manifest["outputs"][name] = file_record(Path(path))
        self._write_manifest()

    def finish(self, *, status: str = "succeeded", failure: dict[str, Any] | None = None) -> None:
        if status not in {"succeeded", "failed"}:
            raise ValueError("status must be 'succeeded' or 'failed'")
        self.manifest["status"] = status
        self.manifest["finished_at_utc"] = _utc_now()
        self.manifest["failure"] = _json_safe(failure)
        self._write_manifest()


def begin_run(
    results_root: Path,
    params: argparse.Namespace | dict[str, Any],
    *,
    run_id: str | None = None,
    workflow: str | None = None,
    return_context: bool = False,
) -> Path | RunContext:
    """Create one immutable run directory and capture resolved parameters.

    Backward compatibility: by default this returns the run directory ``Path`` and
    writes ``params.json`` exactly as earlier callers expect. New runners can pass
    ``return_context=True`` to receive a :class:`RunContext` with manifest helpers.
    """

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_root = Path(results_root)
    run_dir = results_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    raw_params = vars(params) if isinstance(params, argparse.Namespace) else dict(params)
    payload = {
        "schema_version": 1,
        "run_id": resolved_run_id,
        "created_at_utc": _utc_now(),
        "results_root": str(results_root),
        "run_directory": str(run_dir),
        "parameters": _json_safe(raw_params),
    }
    (run_dir / "params.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not return_context:
        return run_dir
    return RunContext(
        run_dir=run_dir,
        workflow=workflow or results_root.name,
        parameters=raw_params,
    )
