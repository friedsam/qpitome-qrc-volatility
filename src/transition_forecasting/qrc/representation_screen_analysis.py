"""Exact HAR helpers required by the canonical Case151 assay."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transition_forecasting.modeling.stage_e_classical_baselines import HAR_FEATURES


def _fit_har(frame: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = frame[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def _parse_mixed_utc(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="raise", utc=True, format="mixed")
    except (TypeError, ValueError):
        parsed: list[pd.Timestamp] = []
        for value in values.astype(str):
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            parsed.append(timestamp)
        return pd.Series(
            pd.DatetimeIndex(parsed), index=values.index, name=values.name
        )


def _prequential_har_residuals(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    *,
    blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = frame[list(HAR_FEATURES)].to_numpy(dtype=float)
    dates = _parse_mixed_utc(frame["origin_date"])
    train_dates = np.asarray(sorted(dates[train_mask].unique()))
    chunks = [
        chunk
        for chunk in np.array_split(train_dates, min(blocks, len(train_dates)))
        if len(chunk)
    ]
    predictions = np.full_like(y, np.nan, dtype=float)
    for block_index in range(1, len(chunks)):
        score_dates = chunks[block_index]
        first_score_date = score_dates[0]
        fit_mask = train_mask & dates.lt(first_score_date).to_numpy()
        score_mask = train_mask & dates.isin(score_dates).to_numpy()
        minimum_fit = max(10, len(HAR_FEATURES) + 2)
        if fit_mask.sum() < minimum_fit or not score_mask.any():
            continue
        scaler = StandardScaler()
        model = Ridge(alpha=100.0)
        model.fit(scaler.fit_transform(x[fit_mask]), y[fit_mask])
        predictions[score_mask] = model.predict(scaler.transform(x[score_mask]))
    valid = train_mask & np.isfinite(predictions).all(axis=1)
    residuals = y - predictions
    return residuals, valid
