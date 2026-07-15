from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_submission.py"
    spec = importlib.util.spec_from_file_location("run_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_command_plan_contains_complete_data_pipeline() -> None:
    runner = load_runner()
    scripts = [command[1] for command in runner.command_plan("data")]
    assert scripts == [
        "scripts/data/download_market_data.py",
        "scripts/data/download_external_datasets.py",
        "scripts/data/inventory_raw_datasets.py",
        "scripts/data/build_volatility_dataset.py",
        "scripts/data/validate_volatility_dataset.py",
        "scripts/data/build_monthly_market_features.py",
    ]


def test_validate_data_plan_does_not_download_or_rebuild() -> None:
    runner = load_runner()
    scripts = [command[1] for command in runner.command_plan("validate-data")]
    assert scripts == [
        "scripts/data/inventory_raw_datasets.py",
        "scripts/data/validate_volatility_dataset.py",
    ]


def test_execute_records_successful_commands_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "VALIDATE_DATA_COMMANDS",
        (("python", "first.py"), ("python", "second.py")),
    )
    monkeypatch.setattr(runner, "REQUIRED_DATA_OUTPUTS", ("data/result.txt",))
    monkeypatch.setattr(runner, "git_value", lambda *args: "test-value")

    output = tmp_path / "data" / "result.txt"
    output.parent.mkdir(parents=True)
    output.write_text("stable output\n", encoding="utf-8")

    calls: list[tuple[str, ...]] = []

    def fake_run_command(argv, run_dir, index):
        calls.append(tuple(argv))
        log = run_dir / f"{index:02d}.log"
        log.write_text("ok\n", encoding="utf-8")
        return runner.CommandRecord(
            argv=list(argv),
            started_at_utc="2026-07-15T00:00:00+00:00",
            finished_at_utc="2026-07-15T00:00:01+00:00",
            duration_seconds=1.0,
            returncode=0,
            log_path=str(log.relative_to(tmp_path)),
        )

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    code, run_dir = runner.execute(
        "validate-data", tmp_path / "results", run_id="test-success"
    )

    assert code == 0
    assert calls == [("python", "first.py"), ("python", "second.py")]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["failure"] is None
    assert len(manifest["commands"]) == 2
    assert manifest["required_outputs"]["data/result.txt"]["exists"] is True
    assert len(manifest["required_outputs"]["data/result.txt"]["sha256"]) == 64


def test_execute_stops_after_first_failed_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "VALIDATE_DATA_COMMANDS",
        (("python", "first.py"), ("python", "second.py")),
    )
    monkeypatch.setattr(runner, "REQUIRED_DATA_OUTPUTS", ())
    monkeypatch.setattr(runner, "git_value", lambda *args: None)

    calls: list[tuple[str, ...]] = []

    def fake_run_command(argv, run_dir, index):
        calls.append(tuple(argv))
        log = run_dir / f"{index:02d}.log"
        log.write_text("failed\n", encoding="utf-8")
        return runner.CommandRecord(
            argv=list(argv),
            started_at_utc="2026-07-15T00:00:00+00:00",
            finished_at_utc="2026-07-15T00:00:01+00:00",
            duration_seconds=1.0,
            returncode=7,
            log_path=str(log.relative_to(tmp_path)),
        )

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    code, run_dir = runner.execute(
        "validate-data", tmp_path / "results", run_id="test-failure"
    )

    assert code == 1
    assert calls == [("python", "first.py")]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["returncode"] == 7
    assert len(manifest["commands"]) == 1


def test_execute_fails_when_required_output_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "VALIDATE_DATA_COMMANDS", ())
    monkeypatch.setattr(runner, "REQUIRED_DATA_OUTPUTS", ("missing.txt",))
    monkeypatch.setattr(runner, "git_value", lambda *args: None)

    code, run_dir = runner.execute(
        "validate-data", tmp_path / "results", run_id="test-missing"
    )

    assert code == 1
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "reason": "missing_required_outputs",
        "paths": ["missing.txt"],
    }
