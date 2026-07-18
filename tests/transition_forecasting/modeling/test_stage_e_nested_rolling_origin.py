from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_nested_rolling_origin.py")
SPEC = importlib.util.spec_from_file_location("stage_e_nested_rolling_origin", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rolling_results() -> pd.DataFrame:
    rows = []
    for fold in (1, 2, 3):
        for model in MODULE.MODELS:
            for alpha, score in ((1.0, 0.9 + 0.01 * fold), (10.0, 0.8 + 0.01 * fold), (100.0, 0.85 + 0.01 * fold)):
                seeds = (0,) if model == "sequence_ridge" else (1, 2, 3)
                for seed in seeds:
                    rows.append({"fold": fold, "model": model, "seed": seed, "alpha": alpha, "val_qlike": score})
    return pd.DataFrame(rows)


def test_nested_alpha_selection_uses_only_earlier_folds() -> None:
    results = _rolling_results()
    selected_fold2 = MODULE.select_nested_alphas(results, 2)
    selected_fold3 = MODULE.select_nested_alphas(results, 3)

    assert selected_fold2 == {model: 10.0 for model in MODULE.MODELS}
    assert selected_fold3 == {model: 10.0 for model in MODULE.MODELS}

    changed = results.copy()
    changed.loc[(changed["fold"] == 3) & (changed["alpha"] == 1.0), "val_qlike"] = -99.0
    assert MODULE.select_nested_alphas(changed, 3) == selected_fold3


def test_nested_alpha_selection_rejects_first_fold() -> None:
    try:
        MODULE.select_nested_alphas(_rolling_results(), 1)
    except ValueError as error:
        assert "earlier fold" in str(error)
    else:
        raise AssertionError("expected nested selection to reject fold 1")
