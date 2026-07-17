"""Shared scoring helpers for held-out binary predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def binary_summary(
    frame: pd.DataFrame,
    model: str,
    *,
    target: str = "y",
    clip: float = 1e-6,
) -> dict[str, float | int | str]:
    """Return the locked row-weighted binary prediction summary."""

    use = frame[[target, model]].dropna()
    y = use[target].to_numpy(int)
    p = np.clip(use[model].to_numpy(float), clip, 1.0 - clip)
    two_class = len(np.unique(y)) == 2
    return {
        "model": model,
        "n": int(len(use)),
        "recovery_rate": float(y.mean()) if len(y) else np.nan,
        "auc": float(roc_auc_score(y, p)) if two_class else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if two_class else np.nan,
        "logloss": float(log_loss(y, p)) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }


def cluster_weighted_summary(
    frame: pd.DataFrame,
    model: str,
    *,
    target: str = "y",
    cluster: str = "cluster_id",
    clip: float = 1e-6,
) -> dict[str, float | int | str]:
    """Return equal-cluster-weighted log-loss and Brier summaries."""

    losses: list[dict[str, float | int | str]] = []
    for cluster_id, group in frame.dropna(subset=[model]).groupby(cluster):
        y = group[target].to_numpy(int)
        p = np.clip(group[model].to_numpy(float), clip, 1.0 - clip)
        losses.append({
            "cluster_id": cluster_id,
            "n": int(len(group)),
            "mean_logloss": float(log_loss(y, p, labels=[0, 1])),
            "mean_brier": float(np.mean((p - y) ** 2)),
        })

    loss = pd.DataFrame(losses)
    if loss.empty:
        return {
            "model": model,
            "n_clusters": 0,
            "cluster_mean_logloss": np.nan,
            "cluster_mean_brier": np.nan,
            "median_cluster_size": np.nan,
        }
    return {
        "model": model,
        "n_clusters": int(loss["cluster_id"].nunique()),
        "cluster_mean_logloss": float(loss["mean_logloss"].mean()),
        "cluster_mean_brier": float(loss["mean_brier"].mean()),
        "median_cluster_size": float(loss["n"].median()),
    }


def proper_score_deltas(
    frame: pd.DataFrame,
    model: str,
    *,
    baseline: str = "D1",
    target: str = "y",
    cluster: str = "cluster_id",
    clip: float = 1e-8,
) -> dict[str, float | int | str]:
    """Compare a model with a baseline using row- and cluster-mean proper scores."""

    use = frame[[target, baseline, model, cluster]].dropna().copy()
    y = use[target].to_numpy(int)
    p0 = np.clip(use[baseline].to_numpy(float), clip, 1.0 - clip)
    p1 = np.clip(use[model].to_numpy(float), clip, 1.0 - clip)

    ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    br0 = (p0 - y) ** 2
    br1 = (p1 - y) ** 2

    use["dll"] = ll1 - ll0
    use["dbr"] = br1 - br0
    by_cluster = use.groupby(cluster)[["dll", "dbr"]].mean()
    return {
        "model": model,
        "n": int(len(use)),
        "n_clusters": int(use[cluster].nunique()),
        "delta_logloss": float(np.mean(ll1 - ll0)),
        "delta_brier": float(np.mean(br1 - br0)),
        "cluster_mean_delta_logloss": float(by_cluster["dll"].mean()),
        "cluster_mean_delta_brier": float(by_cluster["dbr"].mean()),
    }
