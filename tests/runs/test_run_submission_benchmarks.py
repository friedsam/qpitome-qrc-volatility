from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "runs" / "run_submission_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_submission_benchmarks", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_primary_shot_grid_and_final_output_root_are_frozen(tmp_path: Path) -> None:
    runner = load_runner()
    ids = runner.BenchmarkIds("run", "run", "run", "run")
    roots = runner.benchmark_roots(tmp_path / "results" / "runs" / "run")
    command = runner.workflow_commands(
        "palindrome-shots",
        profile="primary",
        ids=ids,
        fold_dir=Path("/fold"),
        mnist_raw_dir=Path("/mnist"),
        roots=roots,
        mnist_shard_root=tmp_path / "shards",
        mnist_shards=8,
        resume=False,
    )[0]
    text = " ".join(command)
    assert "--shot-counts 100 250 500 1000 2000 5000 10000 20000" in text
    assert "--measurement-seeds 20260717" in text
    assert text.endswith("20260726")
    assert f"--out-root {roots['palindrome-shots']}" in text


def test_mnist_plan_uses_literal_shard_ids_and_separate_roots(tmp_path: Path) -> None:
    runner = load_runner()
    commands = runner.mnist_commands(
        profile="smoke",
        run_id="submission_001",
        raw_dir=tmp_path / "raw",
        run_root=tmp_path / "run",
        shard_root=tmp_path / "shards",
        shard_count=2,
        resume=True,
    )
    assert len(commands) == 4
    assert "submission_001_shard_000_of_002" in commands[1]
    assert "submission_001_shard_001_of_002" in commands[2]
    assert "--resume" in commands[1]
    assert str(tmp_path / "shards") in commands[1]
    assert commands[-1][2] == "merge"
    assert "--shard-dirs" in commands[-1]
    assert str(tmp_path / "run") in commands[-1]


def test_final_hierarchy_is_pipeline_shaped(tmp_path: Path) -> None:
    runner = load_runner()
    run_dir = tmp_path / "results" / "runs" / "submission_001"
    roots = runner.benchmark_roots(run_dir)
    assert roots["mnist-palindrome"] == run_dir / "files" / "mnist" / "run"
    assert roots["palindrome-noise"] == run_dir / "files" / "quantum_studies" / "noise" / "run"
    assert roots["palindrome-scaling"] == run_dir / "files" / "quantum_studies" / "scaling" / "run"
    assert roots["palindrome-shots"] == run_dir / "files" / "quantum_studies" / "shots" / "run"


def test_required_inventory_and_validation(tmp_path: Path) -> None:
    runner = load_runner()
    run_dir = tmp_path / "results" / "runs" / "submission_001"
    roots = runner.benchmark_roots(run_dir)
    run = roots["palindrome-noise"] / "submission_001"
    for relative in runner.REQUIRED_RELATIVE_OUTPUTS["palindrome-noise"]:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    assert runner.missing_required_outputs(
        "palindrome-noise", "submission_001", roots
    ) == []
    inventory = runner.required_output_inventory(
        "palindrome-noise", "submission_001", roots
    )
    assert all(item["exists"] for item in inventory.values())
    assert all(len(item["sha256"]) == 64 for item in inventory.values())


def test_validate_only_uses_existing_aggregate_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "git_value", lambda *args: "test")
    run_dir = tmp_path / "results" / "runs" / "validate_001"
    run_dir.mkdir(parents=True)
    roots = runner.benchmark_roots(run_dir)
    for workflow in runner.WORKFLOW_ORDER:
        run = roots[workflow] / "validate_001"
        for relative in runner.REQUIRED_RELATIVE_OUTPUTS[workflow]:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok", encoding="utf-8")

    args = runner.parse_args(
        [
            "validate-existing",
            "--run-id",
            "validate_001",
            "--run-dir",
            str(run_dir),
        ]
    )
    code, observed_run_dir = runner.execute(args)
    assert code == 0
    assert observed_run_dir == run_dir
    metadata = run_dir / "files" / "quantum_studies"
    manifest = json.loads((metadata / "benchmark_manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["commands"] == []
    assert (metadata / "benchmark_artifact_inventory.json").is_file()
