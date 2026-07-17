"""Diagnostic scoring for transition-destination targets."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from transition_destination.core import FEATURES

def feature_auc(episodes: pd.DataFrame, horizon: int, neutral_zone: float, split_date: pd.Timestamp) -> pd.DataFrame:
    target = f"future_return_{horizon}w"
    usable = episodes[episodes[target].notna() & (np.abs(episodes[target]) > neutral_zone)].copy()
    usable["y_positive"] = (usable[target] > neutral_zone).astype(int)
    train, test = usable[usable["date"] < split_date], usable[usable["date"] >= split_date]
    rows = []
    for feature in FEATURES:
        train_auc = roc_auc_score(train["y_positive"], train[feature])
        orientation = 1.0 if train_auc >= 0.5 else -1.0
        holdout_auc = roc_auc_score(test["y_positive"], orientation * test[feature])
        rows.append({"feature": feature, "orientation": "high_positive" if orientation > 0 else "low_positive", "train_n": len(train), "holdout_n": len(test), "holdout_roc_auc": float(holdout_auc)})
    return pd.DataFrame(rows).sort_values("holdout_roc_auc", ascending=False).reset_index(drop=True)
