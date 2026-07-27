from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    PalindromeRealTaskConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_PATH = REPO_ROOT / "config" / "case151" / "expected_metrics.json"
SUPPORT_PATH = REPO_ROOT / "scripts" / "hardware" / "aquila_case151_support.py"
COLLECTOR_PATH = REPO_ROOT / "scripts" / "hardware" / "aquila_case151_collect.py"
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


def test_case151_expected_metric_contract_is_frozen() -> None:
    payload = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    assert payload["canonical_commit"] == "40ec805cc2b4efe416c0a57f1c599cca6def92c3"
    assert payload["source_run"] == "palindrome_real_task_002"
    assert payload["sample_id"] == "P_GE151_^MERV_data_L5"
    assert payload["expected"]["hardware_pooled"]["n_observables"] == 63
    assert payload["expected"]["selected_path"]["qlike"] == 0.7766604033494666
    assert payload["expected"]["selected_transition"]["qlike"] == 1.1057388524991654


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
