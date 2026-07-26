from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_runner() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "runs"
        / "run_submission_benchmarks.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_submission_benchmarks",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_primary_shot_grid_is_frozen() -> None:
    runner = load_runner()
    ids = runner.BenchmarkIds("m", "n", "s", "h")
    command = runner.workflow_commands(
        "palindrome-shots",
        profile="primary",
        ids=ids,
        fold_dir=Path("/fold"),
        mnist_raw_dir=Path("/mnist"),
        mnist_shards=8,
        resume=False,
    )[0]
    text = " ".join(command)
    assert (
        "--shot-counts 100 250 500 1000 2000 5000 10000 20000"
        in text
    )
    assert "--measurement-seeds 20260717" in text
    assert text.endswith("20260726")


def test_mnist_plan_uses_literal_shard_ids() -> None:
    runner = load_runner()
    commands = runner.mnist_commands(
        profile="smoke",
        run_id="mnist_smoke_001",
        raw_dir=Path("/mnist"),
        shard_count=2,
        resume=True,
    )
    assert len(commands) == 4
    assert "mnist_smoke_001_shard_000_of_002" in commands[1]
    assert "mnist_smoke_001_shard_001_of_002" in commands[2]
    assert "--resume" in commands[1]
    assert commands[-1][2] == "merge"
    assert "--shard-dirs" in commands[-1]


def test_required_inventory_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    root = tmp_path / "noise"
    monkeypatch.setitem(
        runner.BENCHMARK_ROOTS,
        "palindrome-noise",
        root,
    )
    run = root / "noise_001"
    for relative in runner.REQUIRED_RELATIVE_OUTPUTS["palindrome-noise"]:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    assert runner.missing_required_outputs(
        "palindrome-noise",
        "noise_001",
    ) == []
    inventory = runner.required_output_inventory(
        "palindrome-noise",
        "noise_001",
    )
    assert all(item["exists"] for item in inventory.values())
    assert all(len(item["sha256"]) == 64 for item in inventory.values())


def test_validate_only_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "git_value", lambda *args: "test")
    run_ids = {
        "mnist-palindrome": "m",
        "palindrome-noise": "n",
        "palindrome-scaling": "s",
        "palindrome-shots": "h",
    }
    for workflow in runner.WORKFLOW_ORDER:
        root = tmp_path / workflow
        monkeypatch.setitem(runner.BENCHMARK_ROOTS, workflow, root)
        for relative in runner.REQUIRED_RELATIVE_OUTPUTS[workflow]:
            path = root / run_ids[workflow] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok", encoding="utf-8")

    args = runner.parse_args(
        [
            "validate-existing",
            "--run-id",
            "validate_001",
            "--results-root",
            str(tmp_path / "runs"),
            "--mnist-run-id",
            "m",
            "--noise-run-id",
            "n",
            "--scaling-run-id",
            "s",
            "--shots-run-id",
            "h",
        ]
    )
    code, run_dir = runner.execute(args)

    assert code == 0
    manifest = json.loads(
        (run_dir / "benchmark_manifest.json").read_text()
    )
    assert manifest["status"] == "succeeded"
    assert manifest["commands"] == []
    assert (run_dir / "benchmark_artifact_inventory.json").is_file()
