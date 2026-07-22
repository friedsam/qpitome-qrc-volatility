from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_MAP_DEVELOPMENT = {
    "har": "har",
    "reference_chain_interaction_off": "chain_interaction_off",
    "reference_chain_interacting_1p00": "chain_interacting_1p00",
    "candidate_ladder_ordered_1p25_symmetric_modes": "frozen_ladder_symmetric_1p25",
}

MODEL_MAP_CONFIRMATION = {
    "har": "har",
    "chain_interaction_off": "chain_interaction_off",
    "chain_interacting_1p00": "chain_interacting_1p00",
    "frozen_ladder_symmetric_1p25": "frozen_ladder_symmetric_1p25",
}

PRIMARY_SCORE = "onset_rise"
MECHANISM_SCORE = "qrc_mean_correction"


@dataclass(frozen=True)
class FrozenPathWarningConfig:
    """Configuration for the development-only forecast-derived warning audit."""

    lead: int = 5
    baseline_horizon: int = 1
    onset_horizon: int = 5
    forecast_horizons: int = 10
    development_folds: tuple[int, ...] = (1, 2, 3)
    confirmation_folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    bootstrap_replicates: int = 10_000
    seed: int = 20260722

    def validate(self) -> None:
        if self.lead < 1:
            raise ValueError("lead must be positive")
        if self.forecast_horizons < 2:
            raise ValueError("forecast_horizons must be at least two")
        if not 1 <= self.baseline_horizon <= self.forecast_horizons:
            raise ValueError("baseline_horizon is outside the forecast path")
        if not 1 <= self.onset_horizon <= self.forecast_horizons:
            raise ValueError("onset_horizon is outside the forecast path")
        if self.onset_horizon <= self.baseline_horizon:
            raise ValueError("onset_horizon must follow baseline_horizon")
        if not self.development_folds or not self.confirmation_folds:
            raise ValueError("development and confirmation folds cannot be empty")
        if set(self.development_folds).intersection(self.confirmation_folds):
            raise ValueError("development and confirmation folds must be disjoint")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")

    @property
    def all_folds(self) -> tuple[int, ...]:
        return self.development_folds + self.confirmation_folds

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_prediction_source(
    run_dir: Path,
    *,
    source: str,
    config: FrozenPathWarningConfig,
) -> pd.DataFrame:
    """Load and normalize one saved prediction artifact."""

    if source not in {"development", "confirmation"}:
        raise ValueError(f"unsupported source: {source}")
    path = Path(run_dir) / "predictions.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "fold",
        "sample_id",
        "lead",
        "label",
        "episode_id",
        "origin_date",
        "model_name",
        "horizon",
        "y_true",
        "y_pred",
        "har_pred",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"prediction artifact is missing columns: {sorted(missing)}")
    if "evaluation_split" in frame.columns:
        unexpected = set(frame["evaluation_split"].dropna().astype(str)) - {"val"}
        if unexpected:
            raise RuntimeError(
                f"warning audit received non-validation rows: {sorted(unexpected)}"
            )

    mapping = (
        MODEL_MAP_DEVELOPMENT if source == "development" else MODEL_MAP_CONFIRMATION
    )
    selected = frame.loc[
        frame["lead"].eq(int(config.lead))
        & frame["model_name"].isin(mapping)
    ].copy()
    selected["model"] = selected["model_name"].map(mapping)
    selected["source"] = source
    selected["fold"] = selected["fold"].astype(int)
    expected_folds = set(
        config.development_folds if source == "development" else config.confirmation_folds
    )
    observed_folds = set(selected["fold"].unique())
    if observed_folds != expected_folds:
        raise RuntimeError(
            f"{source} prediction folds do not match protocol: "
            f"expected={sorted(expected_folds)} observed={sorted(observed_folds)}"
        )
    if selected.empty:
        raise RuntimeError(f"no rows selected from {path}")
    return selected.reset_index(drop=True)


def _linear_slope(values: np.ndarray) -> float:
    x = np.arange(1, len(values) + 1, dtype=float)
    x -= x.mean()
    denominator = float(x @ x)
    return float(values @ x / denominator)


def build_path_score_frame(
    predictions: pd.DataFrame,
    config: FrozenPathWarningConfig,
) -> pd.DataFrame:
    """Collapse ten-horizon forecasts into deterministic warning scores."""

    config.validate()
    required_models = set(MODEL_MAP_CONFIRMATION.values())
    if set(predictions["model"].unique()) != required_models:
        raise RuntimeError("combined predictions do not contain the four frozen models")

    group_columns = [
        "fold",
        "sample_id",
        "label",
        "episode_id",
        "origin_date",
        "model",
        "source",
    ]
    rows: list[dict[str, object]] = []
    expected_horizons = np.arange(1, config.forecast_horizons + 1)
    baseline_index = config.baseline_horizon - 1
    onset_index = config.onset_horizon - 1

    for keys, local in predictions.sort_values("horizon").groupby(
        group_columns,
        sort=False,
    ):
        horizons = local["horizon"].to_numpy(dtype=int)
        if not np.array_equal(horizons, expected_horizons):
            raise RuntimeError(
                f"incomplete or duplicated path for {keys}: {horizons.tolist()}"
            )
        y_true = local["y_true"].to_numpy(dtype=float)
        y_pred = local["y_pred"].to_numpy(dtype=float)
        har_pred = local["har_pred"].to_numpy(dtype=float)
        if not (
            np.isfinite(y_true).all()
            and np.isfinite(y_pred).all()
            and np.isfinite(har_pred).all()
        ):
            raise RuntimeError(f"non-finite path values for {keys}")

        correction = y_pred - har_pred
        row: dict[str, object] = dict(zip(group_columns, keys))
        row.update(
            {
                "onset_rise": float(y_pred[onset_index] - y_pred[baseline_index]),
                "peak_rise": float(y_pred.max() - y_pred[baseline_index]),
                "terminal_rise": float(y_pred[-1] - y_pred[baseline_index]),
                "path_slope": _linear_slope(y_pred),
                "onset_level": float(y_pred[onset_index]),
                "peak_level": float(y_pred.max()),
                "forecast_mean": float(y_pred.mean()),
                "qrc_onset_correction": float(correction[onset_index]),
                "qrc_peak_correction": float(correction.max()),
                "qrc_mean_correction": float(correction.mean()),
                "qrc_correction_slope": _linear_slope(correction),
                "oracle_onset_rise": float(
                    y_true[onset_index] - y_true[baseline_index]
                ),
            }
        )
        for horizon in range(2, config.forecast_horizons + 1):
            index = horizon - 1
            row[f"rise_h{horizon}"] = float(
                y_pred[index] - y_pred[baseline_index]
            )
            row[f"correction_h{horizon}"] = float(correction[index])
        rows.append(row)

    scores = pd.DataFrame(rows)
    model_counts = scores.groupby(["fold", "model"]).size().unstack()
    if model_counts.isna().any().any() or model_counts.nunique(axis=1).ne(1).any():
        raise RuntimeError("frozen models do not have matched sample counts")
    return scores.sort_values(["fold", "sample_id", "model"]).reset_index(drop=True)


def warning_score_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, float]:
    """Evaluate a continuous warning score and its natural zero threshold."""

    labels = frame["label"].to_numpy(dtype=int)
    score = frame[score_column].to_numpy(dtype=float)
    if len(np.unique(labels)) != 2 or not np.isfinite(score).all():
        raise ValueError("warning score metrics require finite values and both labels")
    positive = labels == 1
    control = ~positive
    predicted = score > 0.0
    pooled_std = float(score.std(ddof=1))
    standardized_difference = (
        float((score[positive].mean() - score[control].mean()) / pooled_std)
        if pooled_std > 0
        else 0.0
    )
    return {
        "average_precision": float(average_precision_score(labels, score)),
        "roc_auc": float(roc_auc_score(labels, score)),
        "mean_score_transition": float(score[positive].mean()),
        "mean_score_control": float(score[control].mean()),
        "standardized_mean_difference": standardized_difference,
        "precision_at_zero": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "recall_at_zero": float(recall_score(labels, predicted, zero_division=0)),
        "false_positive_rate_at_zero": float(predicted[control].mean()),
        "positive_rate_at_zero": float(predicted.mean()),
        "samples": int(len(frame)),
        "episodes": int(frame["episode_id"].nunique()),
        "prevalence": float(labels.mean()),
    }


def metric_table(
    scores: pd.DataFrame,
    config: FrozenPathWarningConfig,
    *,
    score_columns: Iterable[str],
) -> pd.DataFrame:
    """Return pooled, period-specific and fold-specific score metrics."""

    periods = {
        "all_folds": config.all_folds,
        "development_folds_1_3": config.development_folds,
        "confirmation_folds_4_8": config.confirmation_folds,
    }
    rows: list[dict[str, object]] = []
    for period, folds in periods.items():
        for model, local_model in scores.loc[scores["fold"].isin(folds)].groupby(
            "model"
        ):
            for score_column in score_columns:
                if model == "har" and score_column.startswith("qrc_"):
                    continue
                rows.append(
                    {
                        "period": period,
                        "fold": 0,
                        "model": model,
                        "score": score_column,
                        **warning_score_metrics(
                            local_model,
                            score_column=score_column,
                        ),
                    }
                )
    for (fold, model), local in scores.groupby(["fold", "model"]):
        for score_column in score_columns:
            if model == "har" and score_column.startswith("qrc_"):
                continue
            rows.append(
                {
                    "period": "single_fold",
                    "fold": int(fold),
                    "model": model,
                    "score": score_column,
                    **warning_score_metrics(local, score_column=score_column),
                }
            )
    return pd.DataFrame(rows)


def horizon_curve_table(
    scores: pd.DataFrame,
    config: FrozenPathWarningConfig,
) -> pd.DataFrame:
    """Evaluate forecast-rise and QRC-correction warning signal by horizon."""

    periods = {
        "all_folds": config.all_folds,
        "development_folds_1_3": config.development_folds,
        "confirmation_folds_4_8": config.confirmation_folds,
    }
    rows: list[dict[str, object]] = []
    for period, folds in periods.items():
        selected = scores.loc[scores["fold"].isin(folds)]
        for model, local in selected.groupby("model"):
            for horizon in range(2, config.forecast_horizons + 1):
                rise_column = f"rise_h{horizon}"
                rows.append(
                    {
                        "period": period,
                        "model": model,
                        "score_family": "forecast_rise_from_h1",
                        "horizon": int(horizon),
                        **warning_score_metrics(local, score_column=rise_column),
                    }
                )
                if model != "har":
                    correction_column = f"correction_h{horizon}"
                    rows.append(
                        {
                            "period": period,
                            "model": model,
                            "score_family": "qrc_correction_over_har",
                            "horizon": int(horizon),
                            **warning_score_metrics(
                                local,
                                score_column=correction_column,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _cluster_sample_indices(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    clusters = frame["episode_id"].astype(str).unique()
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    blocks = [np.flatnonzero(frame["episode_id"].astype(str).eq(cluster)) for cluster in sampled]
    return np.concatenate(blocks)


def paired_cluster_bootstrap_ap(
    scores: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    score_column: str,
    folds: Iterable[int],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Episode-cluster bootstrap for a paired AP difference A minus B."""

    key_columns = ["fold", "sample_id", "label", "episode_id"]
    left = scores.loc[
        scores["model"].eq(model_a) & scores["fold"].isin(tuple(folds)),
        key_columns + [score_column],
    ].rename(columns={score_column: "score_a"})
    right = scores.loc[
        scores["model"].eq(model_b) & scores["fold"].isin(tuple(folds)),
        key_columns + [score_column],
    ].rename(columns={score_column: "score_b"})
    paired = left.merge(right, on=key_columns, how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("paired bootstrap models do not align")
    observed = float(
        average_precision_score(paired["label"], paired["score_a"])
        - average_precision_score(paired["label"], paired["score_b"])
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        selected = paired.iloc[_cluster_sample_indices(paired, rng)]
        if selected["label"].nunique() != 2:
            draws[index] = np.nan
            continue
        draws[index] = (
            average_precision_score(selected["label"], selected["score_a"])
            - average_precision_score(selected["label"], selected["score_b"])
        )
    draws = draws[np.isfinite(draws)]
    return {
        "model_a": model_a,
        "model_b": model_b,
        "score": score_column,
        "folds": [int(value) for value in folds],
        "observed_delta_ap": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_model_a_better": float(np.mean(draws > 0.0)),
        "valid_replicates": int(len(draws)),
    }


def cluster_bootstrap_ap(
    scores: pd.DataFrame,
    *,
    model: str,
    score_column: str,
    folds: Iterable[int],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Episode-cluster bootstrap confidence interval for one AP value."""

    frame = scores.loc[
        scores["model"].eq(model) & scores["fold"].isin(tuple(folds))
    ].reset_index(drop=True)
    observed = float(average_precision_score(frame["label"], frame[score_column]))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        selected = frame.iloc[_cluster_sample_indices(frame, rng)]
        if selected["label"].nunique() != 2:
            draws[index] = np.nan
            continue
        draws[index] = average_precision_score(
            selected["label"],
            selected[score_column],
        )
    draws = draws[np.isfinite(draws)]
    return {
        "model": model,
        "score": score_column,
        "folds": [int(value) for value in folds],
        "observed_ap": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_above_prevalence": float(
            np.mean(draws > frame["label"].mean())
        ),
        "valid_replicates": int(len(draws)),
    }
