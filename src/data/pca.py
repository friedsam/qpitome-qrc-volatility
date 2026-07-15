from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data.features import DEFAULT_FEATURE_COLUMNS, validate_columns


@dataclass
class PCATransformedSplits:
    splits: dict[str, pd.DataFrame]
    feature_columns: list[str]
    scaler: StandardScaler
    pca: PCA
    explained_variance: pd.DataFrame


def fit_transform_pca_splits_train_only(
    splits: dict[str, pd.DataFrame],
    *,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    n_components: int = 6,
    prefix: str = "pca",
) -> PCATransformedSplits:
    """Fit StandardScaler + PCA on train only and transform all splits.

    Returned DataFrames preserve date and target columns, then append PCA columns.
    """
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    target_columns = target_columns or []

    if "train" not in splits:
        raise ValueError("splits must contain a 'train' split")

    for split in splits.values():
        validate_columns(split, feature_columns=feature_columns, target_columns=target_columns)

    scaler = StandardScaler()
    X_train = splits["train"][feature_columns].to_numpy(dtype=float)
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train_scaled)

    pca_columns = [f"{prefix}_{i + 1}" for i in range(n_components)]
    transformed_splits: dict[str, pd.DataFrame] = {}

    keep_columns = ["date", *target_columns]
    keep_columns = [col for col in keep_columns if col in splits["train"].columns]

    for name, split in splits.items():
        X_scaled = scaler.transform(split[feature_columns].to_numpy(dtype=float))
        X_pca = pca.transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=split.index)
        transformed_splits[name] = pd.concat(
            [split[keep_columns].reset_index(drop=True), pca_df.reset_index(drop=True)],
            axis=1,
        )

    explained = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    return PCATransformedSplits(
        splits=transformed_splits,
        feature_columns=pca_columns,
        scaler=scaler,
        pca=pca,
        explained_variance=explained,
    )
