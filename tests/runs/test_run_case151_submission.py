from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/runs/run_case151_submission.py"
SPEC = importlib.util.spec_from_file_location("run_case151_submission", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_qrc_paths_follow_submission_contract(tmp_path: Path) -> None:
    paths = MODULE.qrc_paths(tmp_path, "submission_001")
    assert paths["simulation_run"] == (
        tmp_path / "submission_001" / "files" / "qrc" / "simulation"
        / "run" / "submission_001"
    )


def test_command_uses_literal_shared_run_id(tmp_path: Path) -> None:
    command = MODULE.command_for_run(
        fold_dir=tmp_path / "folds",
        simulation_root=tmp_path / "simulation" / "run",
        run_id="literal_run_007",
    )
    assert command[-2:] == ("--run-id", "literal_run_007")
    assert "run_palindrome_real_task_relevance_assay.py" in command[1]
    assert "submit" not in " ".join(command).lower()


def test_validate_outputs_writes_sha_inventory(tmp_path: Path) -> None:
    for name in MODULE.REQUIRED_OUTPUT_NAMES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    inventory = MODULE.validate_outputs(tmp_path)
    assert set(inventory) == set(MODULE.REQUIRED_OUTPUT_NAMES)
    assert all(item["exists"] for item in inventory.values())
    assert all(len(item["sha256"]) == 64 for item in inventory.values())


def test_execute_rejects_path_like_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path component"):
        MODULE.execute(
            results_root=tmp_path,
            run_id="bad/run",
            fold_dir=tmp_path / "folds",
        )
