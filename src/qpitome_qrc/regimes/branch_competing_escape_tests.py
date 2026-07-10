"""Final pre-freeze tests for branch competing-escape structure.

The module answers four bounded questions:

1. Does downside pressure predict eventual direction beyond current barrier geometry?
2. Does downside pressure modulate total escape timing?
3. Does a duration-dependent Weibull clock beat a constant-rate exponential clock?
4. Are the conclusions stable across a few fixed early landmarks?

All predictive comparisons are episode-prequential. Training episodes are admitted
only after their first-passage event has completed before the test branch date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.regimes.branch_transition_path import add_transition_path_channels


@dataclass(frozen=True)
class CompetingEscapeTestConfig:
    landmarks: tuple[int, ...] = (5, 10, 15)
    min_train_episodes: int = 18
    probability_clip: float = 1e-6
    max_iter: int = 5000


GEOMETRY_FEATURES = ("distance_to_recovery", "distance_to_relapse")
DOWNSIDE_FEATURE = "downside_shock_pressure"


def add_first_passage_panel(
    daily: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build one causal at-risk row per episode-day through first passage."""
    channels = add_transition_path_channels(daily)
    prices = channels["spy_adj_close"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    resolved = events[
        events["resolved_within_followup"]
        & events["event_type"].isin(["recovery", "relapse"])
    ]
    for episode in resolved.itertuples(index=False):
        branch_idx = int(episode.branch_idx)
        event_day = int(episode.event_day)
        branch_price = float(prices[branch_idx])
        upper = float(episode.upper_barrier)
        lower = float(episode.lower_barrier)
        span = upper - lower
        if span <= 0:
            continue
        for day in range(1, event_day + 1):
            idx = branch_idx + day
            cumulative_return = float(prices[idx] / branch_price - 1.0)
            rows.append(
                {
                    "episode_id": int(episode.episode_id),
                    "branch_idx": branch_idx,
                    "branch_date": episode.branch_date,
                    "event_day": event_day,
                    "event_idx": branch_idx + event_day,
                    "event_type": episode.event_type,
                    "day_in_state": day,
                    "event_today": int(day == event_day),
                    "y_recovery": int(episode.event_type == "recovery"),
                    "cumulative_return": cumulative_return,
                    "distance_to_recovery": (upper - cumulative_return) / span,
                    "distance_to_relapse": (cumulative_return - lower) / span,
                    DOWNSIDE_FEATURE: float(channels.iloc[idx][DOWNSIDE_FEATURE]),
                }
            )
    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return panel


def _fit_probability_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    cfg: CompetingEscapeTestConfig,
) -> float:
    if train["y_recovery"].nunique() < 2:
        raise ValueError("Training frame contains one class")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    max_iter=cfg.max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train.loc[:, features], train["y_recovery"])
    p = float(model.predict_proba(test.loc[:, features])[0, 1])
    return float(np.clip(p, cfg.probability_clip, 1.0 - cfg.probability_clip))


def _eligible_train_ids(
    event_table: pd.DataFrame,
    test_branch_idx: int,
) -> set[int]:
    complete = event_table["event_idx"] < test_branch_idx
    return set(event_table.loc[complete, "episode_id"].astype(int))


def prequential_landmark_direction(
    panel: pd.DataFrame,
    config: CompetingEscapeTestConfig | None = None,
) -> pd.DataFrame:
    """Prequential D0/D1/D2 direction predictions at fixed early landmarks."""
    cfg = config or CompetingEscapeTestConfig()
    event_table = panel.drop_duplicates("episode_id")[
        ["episode_id", "branch_idx", "event_idx", "event_type", "y_recovery"]
    ].copy()
    rows: list[dict[str, object]] = []

    for landmark in cfg.landmarks:
        landmark_rows = panel[panel["day_in_state"] == landmark].copy()
        landmark_rows = landmark_rows.sort_values("branch_idx")
        for test in landmark_rows.itertuples(index=False):
            train_ids = _eligible_train_ids(event_table, int(test.branch_idx))
            train = landmark_rows[landmark_rows["episode_id"].isin(train_ids)]
            if len(train) < cfg.min_train_episodes or train["y_recovery"].nunique() < 2:
                continue
            test_frame = landmark_rows[landmark_rows["episode_id"] == test.episode_id]
            prior = float(np.clip(train["y_recovery"].mean(), cfg.probability_clip, 1 - cfg.probability_clip))
            predictions = {
                "D0_prior": prior,
                "D1_geometry": _fit_probability_model(train, test_frame, GEOMETRY_FEATURES, cfg),
                "D2_geometry_plus_downside": _fit_probability_model(
                    train,
                    test_frame,
                    GEOMETRY_FEATURES + (DOWNSIDE_FEATURE,),
                    cfg,
                ),
            }
            for model_name, probability in predictions.items():
                rows.append(
                    {
                        "landmark_day": landmark,
                        "episode_id": int(test.episode_id),
                        "branch_idx": int(test.branch_idx),
                        "event_type": test.event_type,
                        "y_recovery": int(test.y_recovery),
                        "model": model_name,
                        "p_recovery": probability,
                    }
                )
    return pd.DataFrame(rows)


def summarize_direction_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (landmark, model), group in predictions.groupby(["landmark_day", "model"]):
        y = group["y_recovery"].to_numpy(dtype=int)
        p = group["p_recovery"].to_numpy(dtype=float)
        rows.append(
            {
                "landmark_day": int(landmark),
                "model": model,
                "n_oos": len(group),
                "logloss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
                "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _fit_hazard_logit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    cfg: CompetingEscapeTestConfig,
) -> np.ndarray:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    max_iter=cfg.max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train.loc[:, features], train["event_today"])
    p = model.predict_proba(test.loc[:, features])[:, 1]
    return np.clip(p, cfg.probability_clip, 1.0 - cfg.probability_clip)


def prequential_hazard_coupling(
    panel: pd.DataFrame,
    config: CompetingEscapeTestConfig | None = None,
) -> pd.DataFrame:
    """Score H0, duration, downside, and duration+downside hazards per OOS episode."""
    cfg = config or CompetingEscapeTestConfig()
    event_table = panel.drop_duplicates("episode_id")[
        ["episode_id", "branch_idx", "event_idx"]
    ].copy()
    rows: list[dict[str, object]] = []

    for episode in event_table.sort_values("branch_idx").itertuples(index=False):
        train_ids = _eligible_train_ids(event_table, int(episode.branch_idx))
        if len(train_ids) < cfg.min_train_episodes:
            continue
        train = panel[panel["episode_id"].isin(train_ids)].copy()
        test = panel[panel["episode_id"] == episode.episode_id].copy()
        if train["event_today"].nunique() < 2:
            continue
        train["log_day_in_state"] = np.log(train["day_in_state"].astype(float))
        test["log_day_in_state"] = np.log(test["day_in_state"].astype(float))
        base_hazard = float(
            np.clip(train["event_today"].mean(), cfg.probability_clip, 1 - cfg.probability_clip)
        )
        model_probabilities = {
            "H0_constant": np.full(len(test), base_hazard),
            "H1_duration": _fit_hazard_logit(train, test, ("log_day_in_state",), cfg),
            "HP_downside": _fit_hazard_logit(train, test, (DOWNSIDE_FEATURE,), cfg),
            "H1P_duration_plus_downside": _fit_hazard_logit(
                train,
                test,
                ("log_day_in_state", DOWNSIDE_FEATURE),
                cfg,
            ),
        }
        y = test["event_today"].to_numpy(dtype=int)
        for model_name, p in model_probabilities.items():
            episode_loglik = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            rows.append(
                {
                    "episode_id": int(episode.episode_id),
                    "branch_idx": int(episode.branch_idx),
                    "model": model_name,
                    "n_risk_days": len(test),
                    "episode_loglik": episode_loglik,
                    "episode_nll": -episode_loglik,
                }
            )
    return pd.DataFrame(rows)


def summarize_hazard_predictions(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("model", as_index=False)
        .agg(
            n_oos_episodes=("episode_id", "nunique"),
            mean_episode_nll=("episode_nll", "mean"),
            median_episode_nll=("episode_nll", "median"),
            total_loglik=("episode_loglik", "sum"),
        )
        .sort_values("mean_episode_nll")
        .reset_index(drop=True)
    )


def prequential_clock_comparison(
    events: pd.DataFrame,
    config: CompetingEscapeTestConfig | None = None,
) -> pd.DataFrame:
    """Compare exponential T0 with Weibull T1 using episode-prequential log density."""
    cfg = config or CompetingEscapeTestConfig()
    resolved = events[
        events["resolved_within_followup"]
        & events["event_type"].isin(["recovery", "relapse"])
    ].copy()
    resolved["event_idx"] = resolved["branch_idx"].astype(int) + resolved["event_day"].astype(int)
    rows: list[dict[str, object]] = []

    for test in resolved.sort_values("branch_idx").itertuples(index=False):
        train = resolved[resolved["event_idx"] < int(test.branch_idx)]
        if len(train) < cfg.min_train_episodes:
            continue
        times = train["event_day"].to_numpy(dtype=float)
        test_time = float(test.event_day)

        exp_scale = float(np.mean(times))
        exp_logpdf = float(-np.log(exp_scale) - test_time / exp_scale)

        shape, _, scale = weibull_min.fit(times, floc=0.0)
        weibull_logpdf = float(weibull_min.logpdf(test_time, shape, loc=0.0, scale=scale))

        rows.extend(
            [
                {
                    "episode_id": int(test.episode_id),
                    "branch_idx": int(test.branch_idx),
                    "model": "T0_exponential",
                    "event_day": test_time,
                    "loglik": exp_logpdf,
                    "shape": 1.0,
                },
                {
                    "episode_id": int(test.episode_id),
                    "branch_idx": int(test.branch_idx),
                    "model": "T1_weibull",
                    "event_day": test_time,
                    "loglik": weibull_logpdf,
                    "shape": float(shape),
                },
            ]
        )
    return pd.DataFrame(rows)


def summarize_clock_comparison(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("model", as_index=False)
        .agg(
            n_oos_episodes=("episode_id", "nunique"),
            mean_nll=("loglik", lambda x: float(-np.mean(x))),
            total_loglik=("loglik", "sum"),
            median_fitted_shape=("shape", "median"),
        )
        .sort_values("mean_nll")
        .reset_index(drop=True)
    )
