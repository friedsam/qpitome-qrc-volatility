from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.modeling.classical_benchmarks.common import REQUIRED_GROUPS, group_masks, metric_row


def test_groups_do_not_create_control_leads() -> None:
    frame = pd.DataFrame({"label": [1, 1, 1, 0, 0], "lead": [1, 5, 10, 1, 5]})
    masks = group_masks(frame)
    assert tuple(masks) == REQUIRED_GROUPS
    assert masks["L1"].tolist() == [True, False, False, False, False]
    assert masks["Controls"].tolist() == [False, False, False, True, True]
    assert not any("Control" in name and name != "Controls" for name in masks)


def test_metric_row_exact_forecast_has_ideal_mz() -> None:
    y = np.arange(1.0, 21.0).reshape(2, 10)
    row = metric_row(model="exact", group="Pooled", y_true=y, y_pred=y)
    assert row["rmse"] == 0.0
    assert np.isclose(row["qlike"], 0.0)
    assert np.isclose(row["mz_alpha"], 0.0, atol=1e-12)
    assert np.isclose(row["mz_beta"], 1.0)
    assert np.isclose(row["mz_r2"], 1.0)
