from __future__ import annotations

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


def test_case151_scientific_contract_is_frozen() -> None:
    config = PalindromeRealTaskConfig()
    assert config.folds == (4, 5, 6, 7, 8)
    assert config.lead == 5
    assert config.sequence_length == 40
    assert config.ridge_alphas == (0.1, 1.0, 10.0, 100.0, 1000.0)
    assert config.correction_lambdas == (0.0, 0.25, 0.5, 1.0)
    schedule = next(
        item
        for item in CROSSOVER_SCHEDULES
        if item.name == "crossover_Ahalf_B_Ahalf"
    )
    assert schedule.segments == (("A", 0.25), ("B", 0.5), ("A", 0.25))


def test_occupation_pair_bank_has_63_features() -> None:
    probabilities = np.full((2, 3, 64), 1.0 / 64.0)
    banks = build_crossover_feature_banks(probabilities)
    assert banks["occupation_pair_raw"].shape == (2, 63)


def test_expected_metrics_pin_case_and_hardware_identity() -> None:
    path = REPO_ROOT / "config/transition_forecasting/qrc/case151/expected_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["canonical_commit"] == "40ec805cc2b4efe416c0a57f1c599cca6def92c3"
    assert payload["source_run"] == "palindrome_real_task_002"
    assert payload["sample_id"] == "P_GE151_^MERV_data_L5"
    assert set(payload["hardware_job_ids"]) == {"10", "20", "40"}
    assert payload["expected"]["hardware_pooled"]["n_observables"] == 63
