"""Small classical probabilistic baselines for branch resolution.

The initial benchmark is intentionally narrow: recovery versus relapse among
causally extracted branch episodes. Mixed episodes remain in chronological
protocol geometry but are excluded from binary model fitting and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BINARY_OUTCOMES = ("recovery", "relapse")

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "vix_only": ("vix_close",),
    "current_state": (
        "stress_ratio",
        "drawdown_120d",
        "rv_ratio_5_20_branch",
        "return_5d_branch",
    ),
    "state_plus_motion": (
        "stress_ratio",
        "drawdown_120d",
        "rv_ratio_5_20_branch",
        "return_5d_branch",
        "rv_5d_change_5d_branch",
        "worst_return_5d_in_prior_window",
    ),
}


@dataclass(frozen=True)
class BranchLogitConfig:
    c: float = 1.0
    max_iter: int = 5000
    probability_clip: float = 1e-6


def add_branch_baseline_features(
    episodes: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Attach causal baseline features available at each branch point."""
    required_episode = {
        "episode_id",
        "branch_idx",
        "outcome",
        "rv_20d",
        "branch_stress_cut",
        "drawdown_120d",
        "rv_ratio_5_20_branch",
        "rv_5d_change_5d_branch",
        "return_5d_branch",
        "worst_return_5d_in_prior_window",
    }
    missing = required_episode - set(episodes.columns)
    if missing:
        raise KeyError(f"Missing episode features: {sorted(missing)}")
    if "vix_close" not in daily.columns:
        raise KeyError("Daily data missing vix_close")

    out = episodes.copy()
    branch_idx = out["branch_idx"].astype(int).to_numpy()
    if branch_idx.min() < 0 or branch_idx.max() >= len(daily):
        raise IndexError("Episode branch_idx falls outside daily data")

    out["vix_close"] = daily.iloc[branch_idx]["vix_close"].to_numpy(dtype=float)
    out["stress_ratio"] = out["rv_20d"] / out["branch_stress_cut"]
    return out


def binary_episode_frame(episodes: pd.DataFrame) -> pd.DataFrame:
    """Return recovery/relapse episodes with y=1 for recovery."""
    out = episodes[episodes["outcome"].isin(BINARY_OUTCOMES)].copy()
    out["y_recovery"] = (out["outcome"] == "recovery").astype(int)
    return out


def empirical_prior_probability(
    train: pd.DataFrame,
    clip: float = 1e-6,
) -> float:
    """Historical recovery rate among prior binary outcomes."""
    if train.empty:
        raise ValueError("Cannot estimate class prior from empty training set")
    p = float(train["y_recovery"].mean())
    return float(np.clip(p, clip, 1.0 - clip))


def fit_logit_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: tuple[str, ...],
    config: BranchLogitConfig | None = None,
) -> float:
    """Fit train-only standardized logistic regression and predict one episode."""
    cfg = config or BranchLogitConfig()
    if len(test) != 1:
        raise ValueError("Prequential baseline expects exactly one test episode")
    missing = set(feature_columns) - set(train.columns)
    missing |= set(feature_columns) - set(test.columns)
    if missing:
        raise KeyError(f"Missing baseline features: {sorted(missing)}")
    if train["y_recovery"].nunique() < 2:
        raise ValueError("Logistic regression training set contains only one class")

    x_train = train.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train["y_recovery"].to_numpy(dtype=int)
    x_test = test.loc[:, feature_columns].to_numpy(dtype=float)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=cfg.c,
                    max_iter=cfg.max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    p = float(model.predict_proba(x_test)[0, 1])
    return float(np.clip(p, cfg.probability_clip, 1.0 - cfg.probability_clip))
