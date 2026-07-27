from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    PalindromeRealTaskConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_PATH = REPO_ROOT / "config" / "case151" / "expected_metrics.json"
AGENT_SPEC_PATH = REPO_ROOT / "config" / "case151" / "agent_run_spec.json"
SUPPORT_PATH = REPO_ROOT / "scripts" / "hardware" / "aquila_case151_support.py"
COLLECTOR_PATH = REPO_ROOT / "scripts" / "hardware" / "aquila_case151_collect.py"
SIMULATION_RUNNER_PATH = REPO_ROOT / "scripts" / "reproduction" / "run_case151_simulation.py"
REFERENCE_ROOT = REPO_ROOT / "reference" / "case151" / "freeze_001"
REFERENCE_SHA256 = {
    "case151_encoded_sequence.csv": "48359a61b773edf7779ab11e626b75ffae163245dcca6f85f4e402423458e21a",
    "case151_expected_curve.csv": "a12d2eb30f21c869f96caf4699b5d72bbe75310f53932ced2335c7a3185cf161",
    "case151_freeze.json": "a62e229e5ca141b1ef3df5f3f514f0271c7a1a80e271874e7c30887487fae8ef",
    "case151_freeze.npz": "33c602d97ad55bc3e7272caa81fe44f21bca4b151384884662027507f3342201",
    "case151_geometry.csv": "39cd16fb1aebf5e9fdd5e02ca8957398d1c6bc99c7f7cb0f02cf4bf96375358c",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_case151_run(tmp_path: Path, *, path_har_shift: float) -> Path:
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "case151-test"
    run_dir.mkdir()
    identities = {
        "selected_path": ("palindrome_ordered_on", "on", "path"),
        "selected_transition": ("palindrome_ordered_on", "on", "transition"),
        "interaction_off_path": ("palindrome_ordered_off", "off", "path"),
    }
    rows: list[dict[str, object]] = []
    for group, (model, interactions, scope) in identities.items():
        row: dict[str, object] = {
            "model": model,
            "representation": "level_instability",
            "condition": "ordered",
            "interactions": interactions,
            "scope": scope,
        }
        row.update(expected["expected"][group])
        if group == "selected_path":
            row["har_qlike"] = float(row["har_qlike"]) + path_har_shift
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": 8,
                "model": "palindrome_ordered_on",
                "representation": "level_instability",
                "condition": "ordered",
                "interactions": "on",
                "selected_alpha": 0.1,
                "selected_lambda": 0.25,
            }
        ]
    ).to_csv(run_dir / "readout_selections.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "palindrome_ordered_on",
                "representation": "level_instability",
                "condition": "ordered",
                "interactions": "on",
                "feature_width": 63,
            }
        ]
    ).to_csv(run_dir / "feature_diagnostics.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "test_rows_used": 0,
                "qrc_head_fit_intercept": False,
                "folds": [4, 5, 6, 7, 8],
                "lead": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_case151_expected_metric_contract_is_frozen() -> None:
    payload = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    assert payload["canonical_commit"] == "40ec805cc2b4efe416c0a57f1c599cca6def92c3"
    assert payload["source_run"] == "palindrome_real_task_002"
    assert payload["sample_id"] == "P_GE151_^MERV_data_L5"
    assert payload["expected"]["hardware_pooled"]["n_observables"] == 63
    assert payload["expected"]["selected_path"]["qlike"] == 0.7766604033494666
    assert payload["expected"]["selected_transition"]["qlike"] == 1.1057388524991654


def test_agent_run_spec_distinguishes_historical_and_current_folds() -> None:
    payload = json.loads(AGENT_SPEC_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["safety"]["submit_new_hardware_jobs"] is False
    assert payload["safety"]["relabel_current_metrics_as_historical"] is False
    commands = [str(step["command"]) for step in payload["steps"]]
    assert commands
    assert all("reproduce_case151_results.py" not in command for command in commands)
    for command in commands:
        argv = shlex.split(command)
        assert argv[0] == "python"
        assert (REPO_ROOT / argv[1]).is_file(), command
    aggregate = payload["aggregate_integration"]
    assert "--verification-mode current-pipeline" in aggregate["simulation_command"]
    assert "--archive-existing-failed" in aggregate["simulation_command"]
    assert "not relabelled" in aggregate["metric_policy"]
    assert all(step.get("spends_hardware_credits") is False for step in payload["steps"])


def test_case151_model_contract_uses_fold_specific_selection_grid() -> None:
    config = PalindromeRealTaskConfig(
        folds=(4, 5, 6, 7, 8),
        lead=5,
        max_per_class=24,
        ridge_alphas=(0.1, 1.0, 10.0, 100.0, 1000.0),
        correction_lambdas=(0.0, 0.25, 0.5, 1.0),
        schedule_name="crossover_Ahalf_B_Ahalf",
    )
    config.validate()
    assert config.folds == (4, 5, 6, 7, 8)
    assert config.lead == 5
    assert config.max_per_class == 24
    assert config.ridge_alphas == (0.1, 1.0, 10.0, 100.0, 1000.0)
    assert config.correction_lambdas == (0.0, 0.25, 0.5, 1.0)
    schedule = next(item for item in CROSSOVER_SCHEDULES if item.name == config.schedule_name)
    assert schedule.segments == (("A", 0.25), ("B", 0.5), ("A", 0.25))


def test_simulation_runner_freezes_exact_archived_configuration() -> None:
    runner = _load_module("case151_simulation_runner", SIMULATION_RUNNER_PATH)
    objects = runner.canonical_objects()
    config = objects["config"]
    reservoir = objects["reservoir"]
    assert config.folds == (4, 5, 6, 7, 8)
    assert config.max_per_class == 24
    assert config.representations == ("level_only", "level_instability")
    assert config.ridge_alphas == (0.1, 1.0, 10.0, 100.0, 1000.0)
    assert config.correction_lambdas == (0.0, 0.25, 0.5, 1.0)
    assert reservoir.n_atoms == 6
    assert reservoir.step_duration_us == 0.02
    assert reservoir.probe_fractions == (0.25, 0.5, 1.0)
    assert objects["interaction_scale"] == 1.25
    assert runner.DEFAULT_OUTPUT_ROOT.as_posix().endswith("case151_simulation/run")


def test_current_pipeline_records_metric_drift_without_relabelling(tmp_path: Path) -> None:
    runner = _load_module("case151_current_pipeline_runner", SIMULATION_RUNNER_PATH)
    run_dir = _write_synthetic_case151_run(tmp_path, path_har_shift=-0.04)
    report = runner.verify_run(
        run_dir,
        EXPECTED_PATH,
        verification_mode="current-pipeline",
        reference_root=REFERENCE_ROOT,
    )
    assert report["status"] == "verified"
    assert report["verification_mode"] == "current-pipeline"
    assert report["historical_metric_oracle_applied"] is False
    assert report["historical_metric_match"] is False
    assert report["historical_reference_hashes_verified"] is True
    assert report["metric_deltas_vs_historical_reference"]["selected_path"]["har_qlike"] == pytest.approx(-0.04)


def test_historical_oracle_rejects_current_fold_metric_drift(tmp_path: Path) -> None:
    runner = _load_module("case151_historical_oracle_runner", SIMULATION_RUNNER_PATH)
    run_dir = _write_synthetic_case151_run(tmp_path, path_har_shift=-0.04)
    with pytest.raises(RuntimeError, match="selected_path.har_qlike"):
        runner.verify_run(
            run_dir,
            EXPECTED_PATH,
            verification_mode="historical-oracle",
            reference_root=REFERENCE_ROOT,
        )


def test_failed_case151_output_is_archived_before_retry(tmp_path: Path) -> None:
    runner = _load_module("case151_archive_runner", SIMULATION_RUNNER_PATH)
    output_root = tmp_path / "run"
    run_dir = output_root / "same-run-id"
    run_dir.mkdir(parents=True)
    (run_dir / "partial.txt").write_text("preserve me\n", encoding="utf-8")
    archived = runner.archive_existing_failed_output(run_dir, output_root)
    assert archived is not None
    assert not run_dir.exists()
    assert (archived / "partial.txt").read_text(encoding="utf-8") == "preserve me\n"
    assert archived.parent == output_root / "failed_attempts"


def test_occupation_pair_raw_has_63_features() -> None:
    probabilities = np.zeros((2, 3, 64), dtype=float)
    probabilities[:, :, 0] = 1.0
    matrix = build_crossover_feature_banks(probabilities)["occupation_pair_raw"]
    assert matrix.shape == (2, 63)


def test_hardware_tools_are_retrieval_only() -> None:
    prohibited = (
        "create_quantum_task",
        ".run(",
        "--submit",
        "submit_task",
    )
    for path in (SUPPORT_PATH, COLLECTOR_PATH):
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in prohibited), path
    support = SUPPORT_PATH.read_text(encoding="utf-8")
    collector = COLLECTOR_PATH.read_text(encoding="utf-8")
    assert "connect_aquila" in support
    assert "simulate_linear_program" in support
    assert "QbraidJob" in collector
    assert "generated_during_this_run" in collector
    assert "new_hardware_jobs_submitted" in collector
    assert "predictions.json" in collector
    assert "artifact_manifest.json" in collector


def test_case151_reference_freeze_hashes_are_immutable() -> None:
    assert REFERENCE_ROOT.is_dir()
    observed = {path.name for path in REFERENCE_ROOT.iterdir() if path.is_file()}
    assert observed == set(REFERENCE_SHA256)
    for name, expected in REFERENCE_SHA256.items():
        assert _sha256(REFERENCE_ROOT / name) == expected
