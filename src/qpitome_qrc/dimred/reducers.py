from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import IncrementalPCA, KernelPCA, PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ReducerKind = Literal["none", "pca", "incremental_pca", "kernel_pca"]


@dataclass(frozen=True)
class DimReductionConfig:
    """Configuration for reusable dimensionality-reduction pipelines."""

    kind: ReducerKind = "pca"
    n_components: int | float | None = None
    scale: bool = True
    kernel: str = "rbf"
    gamma: float | None = None
    random_state: int = 42


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """No-op transformer for pipeline compatibility."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


def build_reducer(config: DimReductionConfig):
    """Create dimensionality reducer from config."""
    if config.kind == "none":
        return IdentityTransformer()
    if config.kind == "pca":
        return PCA(n_components=config.n_components, random_state=config.random_state)
    if config.kind == "incremental_pca":
        return IncrementalPCA(n_components=config.n_components)
    if config.kind == "kernel_pca":
        return KernelPCA(
            n_components=config.n_components,
            kernel=config.kernel,
            gamma=config.gamma,
            random_state=config.random_state,
        )
    raise ValueError(f"Unknown reducer kind: {config.kind}")


def build_dimred_pipeline(config: DimReductionConfig) -> Pipeline:
    """Build scaler + reducer pipeline. Fit only on training data."""
    steps = []
    if config.scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("reducer", build_reducer(config)))
    return Pipeline(steps)


def pca_explained_variance_table(pipeline: Pipeline) -> object:
    """Return PCA explained variance table if pipeline contains PCA-like reducer."""
    import pandas as pd

    reducer = pipeline.named_steps.get("reducer")
    if not hasattr(reducer, "explained_variance_ratio_"):
        raise ValueError("Reducer does not expose explained_variance_ratio_.")

    ratios = reducer.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": range(1, len(ratios) + 1),
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": ratios.cumsum(),
        }
    )
