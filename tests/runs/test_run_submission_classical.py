from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_runner():
    path = Path(__file__).resolve().parents[2] / "scripts" / "runs" / "run_submission.py"
    spec = importlib.util.spec_from_file_location("submission_runner_classical", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_financial_classical_plan_packages_by_topic():
    runner = load_runner()
    run_dir = Path("results/runs/example")
    commands = runner.command_plan(
        "financial-classical",
        run_dir,
        transition_source_mode="fallback",
        force=False,
    )
    assert len(commands) == 10
    scripts = [command[1] for command in commands]
    assert scripts[-5:] == [
        "scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py",
        "scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py",
        "scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py",
        "scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py",
        "scripts/transition_forecasting/modeling/classical_benchmarks/validate_classical_run.py",
    ]
    paths = runner.transition_paths(run_dir)
    assert paths["classical_root"] == (
        run_dir
        / "files"
        / "transition_forecasting"
        / "modeling"
        / "classical_baselines"
    )
    assert all("esn_tuning" not in " ".join(command) for command in commands)


def test_classical_commands_use_same_generated_dataset_and_run_id():
    runner = load_runner()
    run_dir = Path("results/runs/literal-id")
    commands = runner.classical_commands(run_dir)
    dataset = str(runner.transition_paths(run_dir)["dataset_1d"])
    for command in commands[:3]:
        assert command[command.index("--dataset-root") + 1] == dataset
        assert command[command.index("--run-id") + 1] == "literal-id"
    assert commands[3][commands[3].index("--run-id") + 1] == "literal-id"
    assert commands[4][commands[4].index("--run-id") + 1] == "literal-id"
