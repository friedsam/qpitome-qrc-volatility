from __future__ import annotations

import csv
import json
from pathlib import Path

from transition_forecasting.results.data_pipeline_package import (
    PACKAGE_FILES,
    package_data_pipeline_run,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_packages_successful_transition_run_and_updates_index(tmp_path: Path) -> None:
    run_id = "qbraid-data-20260726T222311Z"
    run_root = tmp_path / "results" / "runs" / run_id
    output_root = tmp_path / "results" / "transition_forecasting" / "data_pipeline" / "run"
    index_path = tmp_path / "docs" / "transition_forecasting" / "result_index.csv"

    required_outputs = {
        "artifact-a": {"exists": True, "size_bytes": 10, "sha256": "abc"},
        "artifact-b": {"exists": True, "size_bytes": 20, "sha256": "def"},
    }
    run_manifest = {
        "schema_version": 3,
        "run_id": run_id,
        "workflow": "transition-data",
        "status": "succeeded",
        "started_at_utc": "2026-07-26T22:23:11+00:00",
        "finished_at_utc": "2026-07-26T22:28:28+00:00",
        "run_directory": f"results/runs/{run_id}",
        "repository": {"commit": "deadbeef", "branch": "stage1-dev"},
        "environment": {"platform": "Linux"},
        "parameters": {"test_evaluated": False},
        "commands": [
            {
                "argv": ["python", "step.py"],
                "started_at_utc": "2026-07-26T22:23:11+00:00",
                "finished_at_utc": "2026-07-26T22:23:12+00:00",
                "duration_seconds": 1.0,
                "returncode": 0,
                "log_path": "logs/01_step.log",
            }
        ],
        "required_outputs": required_outputs,
        "failure": None,
    }
    _write_json(run_root / "run_manifest.json", run_manifest)
    _write_json(
        run_root / PACKAGE_FILES["raw_acquisition_manifest.json"],
        {
            "source_mode_requested": "fallback",
            "source_mode_used": "fallback",
            "authoritative_source_verified": True,
            "fallback_substitution": False,
        },
    )
    _write_json(
        run_root / PACKAGE_FILES["dataset_manifest.json"],
        {
            "counts": {"samples": 4596},
            "transition_summary": {"episodes": 10},
            "stage_d_summary": {"controls": 3},
            "test_evaluated": False,
        },
    )
    _write_json(
        run_root / PACKAGE_FILES["fold_summary.json"],
        {"folds": 8},
    )
    _write_json(
        run_root / PACKAGE_FILES["data_pipeline_audit.json"],
        {"passed": True, "failures": []},
    )
    _write_json(
        run_root / PACKAGE_FILES["data_pipeline_checksums.json"],
        {"passed": True, "missing_files": []},
    )

    report = package_data_pipeline_run(
        run_root=run_root,
        output_root=output_root,
        index_path=index_path,
        experiment_name="global-transition-data-pipeline-fallback-verified",
        platform_name="qBraid Lab",
    )

    destination = output_root / run_id
    assert report["run_id"] == run_id
    assert destination.is_dir()
    assert (destination / "run_summary.json").is_file()
    assert (destination / "package_manifest.json").is_file()
    summary = json.loads((destination / "run_summary.json").read_text())
    assert summary["duration_seconds"] == 317.0
    assert summary["dataset_counts"] == {"samples": 4596}
    assert summary["data_pipeline_audit_passed"] is True
    assert summary["checksum_report_passed"] is True

    with index_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matching_rows = [row for row in rows if row["run_id"] == run_id]
    assert len(matching_rows) == 1
    assert matching_rows[0]["status"] == "verified"
    assert matching_rows[0]["evidence_path"].endswith(run_id)
