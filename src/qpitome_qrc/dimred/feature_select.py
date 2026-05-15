from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold


@dataclass(frozen=True)
class FeatureSet:
    """Named feature list for controlled experiments."""

    name: str
    features: list[str]


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop highly correlated features using train-set absolute correlation.

    The first feature in each correlated pair is kept; later features are removed.
    This is intentionally simple and deterministic for diagnostic baselines.
    """

    def __init__(self, threshold: float = 0.95, feature_names: list[str] | None = None):
        self.threshold = threshold
        self.feature_names = feature_names

    def fit(self, X, y=None):
        X_df = self._to_frame(X)
        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_columns_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        self.keep_columns_ = [c for c in X_df.columns if c not in self.drop_columns_]
        self.feature_names_out_ = np.asarray(self.keep_columns_)
        return self

    def transform(self, X):
        X_df = self._to_frame(X)
        return X_df[self.keep_columns_].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if self.feature_names is not None:
            return pd.DataFrame(X, columns=self.feature_names)
        return pd.DataFrame(X)


def variance_filter(threshold: float = 0.0) -> VarianceThreshold:
    """Return sklearn VarianceThreshold transformer."""
    return VarianceThreshold(threshold=threshold)


def mutual_information_selector(k: int | str = "all", random_state: int = 42) -> SelectKBest:
    """Return mutual-information feature selector for classification."""
    return SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=random_state),
        k=k,
    )


def correlation_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return pairwise absolute feature correlations sorted descending."""
    corr = df[features].corr().abs()
    rows = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1 :]:
            rows.append({"feature_1": f1, "feature_2": f2, "abs_corr": corr.loc[f1, f2]})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
