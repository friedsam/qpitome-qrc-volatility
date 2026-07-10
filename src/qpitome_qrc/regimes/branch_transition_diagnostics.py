"""Diagnostics for task-aligned branch-transition path representations.

The diagnostics answer two admission questions before any new reservoir is built:

1. Are simple summaries of the proposed path nearly reconstructible from the
   existing ``state_plus_motion`` endpoint baseline?
2. Do explicitly order-sensitive summaries change when temporal ordering is
   destroyed while contemporaneous cross-channel tuples are preserved?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.branch_probabilistic import FEATURE_SETS
from qpitome_qrc.regimes.branch_transition_path import TRANSITION_PATH_COLUMNS


STATE_MOTION_COLUMNS = FEATURE_SETS["state_plus_motion"]
ORDER_INSENSITIVE_SUMMARIES = ("mean", "std")
ORDER_SENSITIVE_SUMMARIES = (
    "last",
    "late_minus_early",
    "slope",
    "lag1_autocorr",
    "late_extreme_fraction",
)


def _safe_lag1_autocorr(x: np.ndarray) -> float:
    left = np.asarray(x[:-1], dtype=float)
    right = np.asarray(x[1:], dtype=float)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def summarize_transition_windows(
    episode_ids: np.ndarray,
    windows: np.ndarray,
) -> pd.DataFrame:
    """Create fixed endpoint, shape, persistence, and recurrence summaries."""
    ids = np.asarray(episode_ids, dtype=int)
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected windows with shape (episode, time, channel); got {values.shape}")
    if len(ids) != len(values):
        raise ValueError("Episode-id and window counts differ")
    if values.shape[2] != len(TRANSITION_PATH_COLUMNS):
        raise ValueError(
            f"Expected {len(TRANSITION_PATH_COLUMNS)} channels; got {values.shape[2]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Transition windows contain non-finite values")

    n_time = values.shape[1]
    split = n_time // 2
    time = np.linspace(-1.0, 1.0, n_time)
    time_denom = float(np.dot(time, time))
    late_start = max(0, n_time - 10)

    rows: list[dict[str, float | int]] = []
    for episode_row, episode_id in enumerate(ids):
        row: dict[str, float | int] = {"episode_id": int(episode_id)}
        for channel_idx, channel in enumerate(TRANSITION_PATH_COLUMNS):
            x = values[episode_row, :, channel_idx]
            x_centered = x - x.mean()
            threshold = float(np.quantile(np.abs(x), 0.75))
            late_extreme_fraction = float(
                np.mean(np.abs(x[late_start:]) >= threshold)
            )
            row[f"{channel}__mean"] = float(x.mean())
            row[f"{channel}__std"] = float(x.std())
            row[f"{channel}__last"] = float(x[-1])
            row[f"{channel}__late_minus_early"] = float(
                x[split:].mean() - x[:split].mean()
            )
            row[f"{channel}__slope"] = float(np.dot(time, x_centered) / time_denom)
            row[f"{channel}__lag1_autocorr"] = _safe_lag1_autocorr(x)
            row[f"{channel}__late_extreme_fraction"] = late_extreme_fraction
        rows.append(row)
    return pd.DataFrame(rows)


def permute_transition_windows(
    windows: np.ndarray,
    *,
    seed: int,
    block_size: int = 1,
) -> np.ndarray:
    """Destroy episode-level temporal order while preserving channel tuples.

    The same time permutation is applied to all channels within an episode, so
    contemporaneous cross-channel relationships are retained. ``block_size=1``
    gives full permutation. Larger blocks preserve within-block local order.
    """
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected 3D windows; got {values.shape}")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    rng = np.random.default_rng(seed)
    out = values.copy()
    n_time = values.shape[1]
    starts = np.arange(0, n_time, block_size)

    for episode_idx in range(len(values)):
        order = rng.permutation(len(starts))
        indices = np.concatenate(
            [
                np.arange(starts[i], min(starts[i] + block_size, n_time))
                for i in order
            ]
        )
        out[episode_idx] = values[episode_idx, indices]
    return out


def redundancy_audit(
    episode_features: pd.DataFrame,
    path_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify how well baseline endpoint features reconstruct path summaries."""
    baseline = episode_features.copy()
    if "stress_ratio" not in baseline.columns:
        baseline["stress_ratio"] = baseline["rv_20d"] / baseline["branch_stress_cut"]

    missing = set(STATE_MOTION_COLUMNS) - set(baseline.columns)
    if missing:
        raise KeyError(f"Missing state_plus_motion features: {sorted(missing)}")

    merged = path_summaries.merge(
        baseline.loc[:, ["episode_id", *STATE_MOTION_COLUMNS]],
        on="episode_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(path_summaries):
        raise ValueError("Not all path summaries matched episode baseline features")

    x = merged.loc[:, STATE_MOTION_COLUMNS].to_numpy(dtype=float)
    loo = LeaveOneOut()
    rows: list[dict[str, float | str]] = []

    for column in path_summaries.columns:
        if column == "episode_id":
            continue
        y = merged[column].to_numpy(dtype=float)
        y_variance = float(np.var(y))

        pearson = merged.loc[:, [*STATE_MOTION_COLUMNS, column]].corr(
            method="pearson"
        )[column].drop(index=column)
        spearman = merged.loc[:, [*STATE_MOTION_COLUMNS, column]].corr(
            method="spearman"
        )[column].drop(index=column)

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        )
        prediction = cross_val_predict(model, x, y, cv=loo)
        if y_variance <= 1e-15:
            loo_r2 = np.nan
        else:
            loo_r2 = 1.0 - float(np.mean((y - prediction) ** 2)) / y_variance

        rows.append(
            {
                "summary": column,
                "summary_type": column.rsplit("__", 1)[-1],
                "max_abs_pearson_with_baseline": float(pearson.abs().max()),
                "max_abs_spearman_with_baseline": float(spearman.abs().max()),
                "loo_ridge_r2_from_baseline": loo_r2,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["loo_ridge_r2_from_baseline", "max_abs_spearman_with_baseline"],
        ascending=False,
    )


def order_destruction_audit(
    original: pd.DataFrame,
    permuted: pd.DataFrame,
    *,
    permutation_name: str,
) -> pd.DataFrame:
    """Measure how strongly each summary changes after temporal permutation."""
    merged = original.merge(
        permuted,
        on="episode_id",
        suffixes=("__original", "__permuted"),
        validate="one_to_one",
    )
    rows: list[dict[str, float | str]] = []

    for column in original.columns:
        if column == "episode_id":
            continue
        a = merged[f"{column}__original"].to_numpy(dtype=float)
        b = merged[f"{column}__permuted"].to_numpy(dtype=float)
        scale = float(np.std(a))
        mean_abs_change = float(np.mean(np.abs(a - b)))
        normalized_change = mean_abs_change / scale if scale > 1e-12 else np.nan
        rho = spearmanr(a, b).statistic
        rows.append(
            {
                "permutation": permutation_name,
                "summary": column,
                "summary_type": column.rsplit("__", 1)[-1],
                "mean_abs_change": mean_abs_change,
                "mean_abs_change_over_original_sd": normalized_change,
                "spearman_original_vs_permuted": float(rho),
            }
        )

    return pd.DataFrame(rows)
