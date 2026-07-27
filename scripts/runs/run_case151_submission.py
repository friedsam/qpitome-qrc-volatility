#!/usr/bin/env python3
"""Run the canonical Case151 QRC stage inside one submission aggregate run.

This focused entry point avoids modifying the shared submission orchestrator while it
is being maintained in parallel. It writes only below
``results/runs/<RUN_ID>/files/qrc/simulation/run/<RUN_ID>`` and the aggregate log
folder. It never submits quantum hardware jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "runs"
DEFAULT_FOLD_DIR = (
    REPO_ROOT / "data" / "processed" / "global_transition_dataset_1d"
    / "purged_walk_forward_folds"
)
CANONICAL_RUNNER = (
    REPO_ROOT / "scripts" / "transition_forecasting" / "qrc"
    / "run_palindrome_real_task_relevance_assay.py"
)
REQUIRED_OUTPUT_NAMES = (
    "params.json",
    "prediction_cells.csv.gz",
    "fold_metrics.csv",
    "pooled_metrics.csv",
    "readout_selections.csv",
    "readout_candidates.csv.gz",
    "feature_diagnostics.csv",
    "baseline_health.csv",
    "channel_scalers.csv",
    "simulation_metadata.csv",
    "summary.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qrc_paths(results_root: Path, run_id: str) -> dict[str, Path]:
    aggregate = Path(results_root) / run_id
    simulation_root = aggregate / "files" / "qrc" / "simulation" / "run"
    return {
        "aggregate": aggregate,
        "logs": aggregate / "logs",
        "simulation_root": simulation_root,
        "simulation_run": simulation_root / run_id,
    }


def command_for_run(*, fold_dir: Path, simulation_root: Path, run_id: str) -> tuple[str, ...]:
    return (
        sys.executable,
        str(CANONICAL_RUNNER.relative_to(REPO_ROOT)),
        "--fold-dir", str(fold_dir),
        "--out-root", str(simulation_root),
        "--run-id", run_id,
    )


def validate_outputs(run_dir: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for name in REQUIRED_OUTPUT_NAMES:
        path = run_dir / name
        item: dict[str, object] = {"exists": path.is_file()}
        if path.is_file():
            item.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        else:
            missing.append(name)
        inventory[name] = item
    if missing:
        raise RuntimeError(f"Case151 run is missing required outputs: {missing}")
    return inventory


def execute(*, results_root: Path, run_id: str, fold_dir: Path) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be one nonempty path component")
    paths = qrc_paths(results_root, run_id)
    if paths["simulation_run"].exists():
        raise FileExistsError(paths["simulation_run"])
    if not fold_dir.is_dir():
        raise FileNotFoundError(fold_dir)
    for name in ("rematched_rolling_manifest.csv", "rematched_rolling_tensors.npz"):
        if not (fold_dir / name).is_file():
            raise FileNotFoundError(fold_dir / name)

    paths["logs"].mkdir(parents=True, exist_ok=True)
    paths["simulation_root"].mkdir(parents=True, exist_ok=True)
    log_path = paths["logs"] / "qrc_case151.log"
    command = command_for_run(
        fold_dir=fold_dir,
        simulation_root=paths["simulation_root"],
        run_id=run_id,
    )
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"canonical Case151 runner failed with {completed.returncode}; see {log_path}"
        )

    inventory = validate_outputs(paths["simulation_run"])
    manifest = {
        "schema_version": 1,
        "stage": "qrc-simulation-case151",
        "run_id": run_id,
        "status": "succeeded",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "canonical_model": {
            "source_commit": "40ec805cc2b4efe416c0a57f1c599cca6def92c3",
            "source_run": "palindrome_real_task_002",
            "model": "palindrome_ordered_on",
            "representation": "level_instability",
            "feature_bank": "occupation_pair_raw",
            "feature_width": 63,
            "fit_intercept": False,
            "fold_specific_readout_selection": True,
            "case151_fold": 8,
            "case151_alpha": 0.1,
            "case151_lambda": 0.25,
            "test_rows_used": 0,
        },
        "command": list(command),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "output_directory": str(paths["simulation_run"].relative_to(REPO_ROOT)),
        "artifacts": inventory,
        "hardware_submission_available": False,
    }
    (paths["simulation_run"] / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths["simulation_run"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical 63-feature Case151 QRC submission stage"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = execute(
        results_root=args.results_root,
        run_id=str(args.run_id),
        fold_dir=Path(args.fold_dir),
    )
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
