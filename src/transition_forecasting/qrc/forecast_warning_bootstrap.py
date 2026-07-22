from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _fast_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ranked = np.asarray(labels, dtype=int)[order]
    positives = int(ranked.sum())
    if positives <= 0:
        return float("nan")
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, len(ranked) + 1, dtype=float)
    return float(precision[ranked == 1].sum() / positives)


def _cluster_index_blocks(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    episode = frame["episode_id"].astype(str).to_numpy()
    return tuple(np.flatnonzero(episode == cluster) for cluster in np.unique(episode))


def _sample_cluster_rows(
    blocks: tuple[np.ndarray, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[int(index)] for index in sampled])


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

    labels = paired["label"].to_numpy(dtype=int)
    score_a = paired["score_a"].to_numpy(dtype=float)
    score_b = paired["score_b"].to_numpy(dtype=float)
    observed = float(
        average_precision_score(labels, score_a)
        - average_precision_score(labels, score_b)
    )
    blocks = _cluster_index_blocks(paired)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=float)
    valid = 0
    for _ in range(int(replicates)):
        selected = _sample_cluster_rows(blocks, rng)
        sampled_labels = labels[selected]
        if np.unique(sampled_labels).size != 2:
            continue
        draws[valid] = (
            _fast_average_precision(sampled_labels, score_a[selected])
            - _fast_average_precision(sampled_labels, score_b[selected])
        )
        valid += 1
    draws = draws[:valid]
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
    labels = frame["label"].to_numpy(dtype=int)
    score = frame[score_column].to_numpy(dtype=float)
    observed = float(average_precision_score(labels, score))
    blocks = _cluster_index_blocks(frame)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=float)
    valid = 0
    for _ in range(int(replicates)):
        selected = _sample_cluster_rows(blocks, rng)
        sampled_labels = labels[selected]
        if np.unique(sampled_labels).size != 2:
            continue
        draws[valid] = _fast_average_precision(sampled_labels, score[selected])
        valid += 1
    draws = draws[:valid]
    return {
        "model": model,
        "score": score_column,
        "folds": [int(value) for value in folds],
        "observed_ap": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_above_prevalence": float(np.mean(draws > labels.mean())),
        "valid_replicates": int(len(draws)),
    }
