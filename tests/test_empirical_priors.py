from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.priors import predict_empirical_prior


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empirical_prior_uses_group_and_falls_back_when_sparse() -> None:
    train = pd.DataFrame({
        "market_key": ["a", "a", "a", "b", "b", "c"],
        "y_recovery": [1, 1, 0, 0, 0, 1],
    })

    pooled = predict_empirical_prior(train, target="y_recovery")
    market_a = predict_empirical_prior(
        train,
        target="y_recovery",
        group_column="market_key",
        group_value="a",
        min_group_rows=3,
    )
    sparse_b = predict_empirical_prior(
        train,
        target="y_recovery",
        group_column="market_key",
        group_value="b",
        min_group_rows=3,
    )

    np.testing.assert_allclose(pooled, 0.5)
    np.testing.assert_allclose(market_a, 2.0 / 3.0)
    np.testing.assert_allclose(sparse_b, pooled)


def test_empirical_prior_clips_extreme_rates() -> None:
    all_positive = pd.DataFrame({"y": [1, 1, 1]})
    all_negative = pd.DataFrame({"y": [0, 0, 0]})

    np.testing.assert_allclose(
        predict_empirical_prior(all_positive, target="y", clip=1e-6),
        1.0 - 1e-6,
    )
    np.testing.assert_allclose(
        predict_empirical_prior(all_negative, target="y", clip=1e-6),
        1e-6,
    )


def test_confirmatory_script_reexports_empirical_prior_helper() -> None:
    module = load_script(
        "confirmatory_empirical_prior",
        "scripts/modeling/run_cross_market_day5_confirmatory_baselines.py",
    )

    assert module.predict_empirical_prior is predict_empirical_prior
