from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.control_strata import CALM, HARD_NEGATIVE, TRANSITION
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc import transition_signal_broad_har as module
from transition_forecasting.qrc.ladder_spacing_broad_har_reanalysis import (
    _load_legacy_spacing_probabilities,
)
from transition_forecasting.qrc.transition_signal_readout_assay import (
    TransitionSignalAssayConfig,
)


def _frame() -> pd.DataFrame:
    rows = []
    for split, offset in (("train", 0), ("val", 3)):
        for index, stratum in enumerate((TRANSITION, CALM, HARD_NEGATIVE)):
            row = {
                "fold": 4,
                "fold_split": split,
                "sample_id": f"{split}_{stratum}",
                "lead": (1, 5, 10)[index],
                "label": int(stratum == TRANSITION),
                "evaluation_stratum": stratum,
                "episode_id": f"E{offset + index}",
                "origin_date": f"2000-01-{offset + index + 1:02d}",
            }
            row.update({name: 0.0 for name in TARGET_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_broad_har_uses_all_training_rows(monkeypatch) -> None:
    frame = _frame()
    y_shape = (len(frame), len(TARGET_COLUMNS))
    seen: dict[str, int] = {}

    def fake_fit_har(frame_arg, y_arg, train_arg):
        seen["har_train"] = int(np.asarray(train_arg, dtype=bool).sum())
        return np.zeros(y_shape, dtype=float)

    def fake_residuals(frame_arg, y_arg, train_arg, *, blocks):
        seen["residual_train"] = int(np.asarray(train_arg, dtype=bool).sum())
        return np.zeros(y_shape, dtype=float), np.asarray(train_arg, dtype=bool)

    def fake_select(matrix, **kwargs):
        correction = np.zeros(y_shape, dtype=float)
        diagnostics = {
            "ridge_alpha": 100.0,
            "transition_weight": 1.0,
            "fit_intercept": False,
            "early_lambda": 0.0,
            "late_lambda": 0.0,
            "eligible": True,
        }
        return diagnostics, pd.DataFrame([diagnostics]), correction, matrix.shape[1]

    monkeypatch.setattr(module, "_fit_har", fake_fit_har)
    monkeypatch.setattr(module, "_prequential_har_residuals", fake_residuals)
    monkeypatch.setattr(module, "select_transition_signal_configuration", fake_select)

    module.evaluate_exact_matrix_broad_har(
        np.zeros((len(frame), 2), dtype=float),
        frame=frame,
        config=TransitionSignalAssayConfig(),
        model_name="test",
        readout_kind="linear",
    )

    assert seen == {"har_train": 3, "residual_train": 3}


def test_legacy_spacing_cache_loads_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "legacy_spacing.npz"
    probabilities = np.full((3, 2, 64), 1.0 / 64.0, dtype=float)
    leads = np.asarray([1, 5, 10], dtype=int)
    np.savez_compressed(
        path,
        probabilities=probabilities,
        lead=leads,
        sample_id=np.asarray(["a", "b", "c"], dtype=object),
        fold_split=np.asarray(["train", "train", "val"], dtype=object),
    )

    loaded = _load_legacy_spacing_probabilities(path, expected_leads=leads)

    assert np.array_equal(loaded, probabilities)
