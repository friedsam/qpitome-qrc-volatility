"""Shared run-directory creation and parameter capture for experiment scripts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def begin_run(
    results_root: Path,
    params: argparse.Namespace | dict[str, Any],
    *,
    run_id: str | None = None,
) -> Path:
    """Create one immutable run directory and write its resolved parameters.

    ``results_root`` is the script-level destination, for example
    ``results/baselines/lstm/run_phase3_lstm_walkforward``. The returned path is
    ``results_root/<run-id>``.
    """

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(results_root) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    raw_params = vars(params) if isinstance(params, argparse.Namespace) else dict(params)
    payload = {
        "schema_version": 1,
        "run_id": resolved_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "run_directory": str(run_dir),
        "parameters": _json_safe(raw_params),
    }
    (run_dir / "params.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # Selection tables are common first-class artifacts across the experiment
    # suite. Creating the directory centrally avoids assay-specific write-order
    # failures while preserving immutable run-root semantics.
    (run_dir / "selection_candidates").mkdir()
    return run_dir
