import numpy as np
import pandas as pd
import pytest

from qpitome_qrc.data.targets import (
    RV_INNOVATION_TARGET,
    add_rv_innovation_target,
    reconstruct_future_rv,
)


def test_add_rv_innovation_target_matches_log_ratio_and_preserves_missing():
    frame = pd.DataFrame(
        {
            "future_rv_20d": [0.20, 0.10, np.nan],
            "rv_20d": [0.10, 0.20, 0.15],
        }
    )

    result = add_rv_innovation_target(frame)

    np.testing.assert_allclose(
        result[RV_INNOVATION_TARGET].iloc[:2].to_numpy(),
        np.log([2.0, 0.5]),
    )
    assert np.isnan(result[RV_INNOVATION_TARGET].iloc[2])
    assert RV_INNOVATION_TARGET not in frame.columns


def test_reconstruct_future_rv_inverts_innovation_target():
    reference = np.array([0.10, 0.20])
    innovation = np.log(np.array([2.0, 0.5]))

    reconstructed = reconstruct_future_rv(reference, innovation)

    np.testing.assert_allclose(reconstructed, np.array([0.20, 0.10]))


def test_add_rv_innovation_target_rejects_nonpositive_finite_values():
    frame = pd.DataFrame(
        {
            "future_rv_20d": [0.20, 0.00],
            "rv_20d": [0.10, 0.20],
        }
    )

    with pytest.raises(ValueError, match="strictly positive"):
        add_rv_innovation_target(frame)
