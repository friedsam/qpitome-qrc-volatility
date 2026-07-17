"""Strictly historical cross-fitting helpers for prequential evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.logistic_offset import logit
from qpitome_qrc.day5.protocol import D1
from qpitome_qrc.evaluation.binary import logistic_pipeline


def historical_crossfit_d1_logits(
    train: pd.DataFrame,
    *,
    min_train: int = 10,
    C: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row positions and strictly historical out-of-fold D1 logits.

    Each cluster-start group is predicted only from rows whose landmark date is
    earlier than that group's cluster start.
    """

    logits = np.full(len(train), np.nan, dtype=float)
    for cluster_start, group in train.groupby("cluster_start", sort=True):
        valid_positions = train.index.get_indexer(group.index)
        prior = train[train["landmark_date"] < cluster_start]
        if len(prior) < min_train or prior["y_recovery"].nunique() < 2:
            continue
        model = logistic_pipeline(C)
        model.fit(prior[D1].to_numpy(float), prior["y_recovery"].to_numpy(int))
        probabilities = model.predict_proba(group[D1].to_numpy(float))[:, 1]
        logits[valid_positions] = logit(probabilities)

    valid = np.isfinite(logits)
    return np.flatnonzero(valid), logits[valid]
