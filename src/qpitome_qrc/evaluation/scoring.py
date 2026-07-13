"""Shared proper-score comparison helpers for held-out predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
