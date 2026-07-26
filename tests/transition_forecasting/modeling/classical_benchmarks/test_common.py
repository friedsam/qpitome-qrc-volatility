import numpy as np
import pandas as pd

from transition_forecasting.modeling.classical_benchmarks.common import (
    REQUIRED_GROUPS,
    group_masks,
    metric_row,
    qlike_loss,
)


def test_groups_do_not_create_control_leads():
    frame = pd.DataFrame(
        {"label": [1, 1, 1, 0, 0], "lead": [1, 5, 10, 1, 5]}
    )
    masks = group_masks(frame)
    assert tuple(masks) == REQUIRED_GROUPS
    assert masks["Controls"].sum() == 2
    assert masks["L1"].sum() == 1
    assert not (masks["L1"] & masks["Controls"]).any()


def test_qlike_zero_for_exact_log_volatility_forecast():
    observed = np.array([[-2.0, -1.5]])
    assert np.allclose(qlike_loss(observed, observed), 0.0)
    row = metric_row(
        model="x",
        group="Pooled",
        y_true=observed,
        y_pred=observed,
    )
    assert row["rmse"] == 0.0
    assert abs(row["qlike"]) < 1e-12
