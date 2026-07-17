from __future__ import annotations

import json
from pathlib import Path

from experiments.runs import RunContext, begin_run, sha256


def test_begin_run_preserves_path_api(tmp_path: Path) -> None:
    run_dir = begin_run(tmp_path / "results", {"alpha": 1}, run_id="run-1")
    assert isinstance(run_dir, Path)
    assert run_dir == tmp_path / "results" / "run-1"
    payload = json.loads((run_dir / "params.json").read_text())
    assert payload["parameters"] == {"alpha": 1}


def test_run_context_records_inputs_outputs_and_completion(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("input\n")
    context = begin_run(
        tmp_path / "results",
        {"beta": 2},
        run_id="run-2",
        workflow="test.workflow",
        return_context=True,
    )
    assert isinstance(context, RunContext)
    context.record_input("source", input_path)
    output_path = context.run_dir / "output.txt"
    output_path.write_text("output\n")
    context.record_output("result", output_path)
    context.finish()

    manifest = json.loads(context.manifest_path.read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["workflow"] == "test.workflow"
    assert manifest["inputs"]["source"]["sha256"] == sha256(input_path)
    assert manifest["outputs"]["result"]["exists"] is True
    assert manifest["finished_at_utc"] is not None


def test_begin_run_refuses_overwrite(tmp_path: Path) -> None:
    begin_run(tmp_path / "results", {}, run_id="same")
    try:
        begin_run(tmp_path / "results", {}, run_id="same")
    except FileExistsError:
        pass
    else:
        raise AssertionError("begin_run must not overwrite an existing run")
