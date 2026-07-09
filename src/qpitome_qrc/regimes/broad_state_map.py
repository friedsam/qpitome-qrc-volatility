"""Causal broad-market regime map for the Phase 3 regime front.

The map is intentionally simple and explicit. It formalizes the provisional
states described in ``docs/experiments/regime_branching_analysis.md`` without
using VIX or future targets.

This is a development taxonomy, not a claim that six latent market regimes are
uniquely true. Threshold sensitivity and historical morphology must be audited
before the map becomes canonical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REGIME_ORDER = (
    "calm",
    "mild_stress",
    "deterioration",
    "acute_crisis",
    "volatile_rebound",
    "easing_normalization",
)


@dataclass(frozen=True)
class BroadRegimeConfig:
    """Thresholds for the provisional causal broad-state map."""

    expanding_min_periods: int = 504
    stress_quantile: float = 0.70
    high_stress_quantile: float = 0.90
    damage_threshold: float = -0.05
    severe_damage_threshold: float = -0.10
    rebound_5d_threshold: float = 0.01
    deterioration_5d_threshold: float = -0.01


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal trailing features required by the broad-state map."""
    required = {
        "date",
        "spy_adj_close",
        "spy_log_return",
        "rv_5d",
        "rv_20d",
        "rv_60d",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing regime inputs: {sorted(missing)}")

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["return_5d"] = out["spy_log_return"].rolling(5).sum()
    out["return_20d"] = out["spy_log_return"].rolling(20).sum()
    out["rv_5d_change_5d"] = out["rv_5d"] - out["rv_5d"].shift(5)
    out["rv_20d_change_5d"] = out["rv_20d"] - out["rv_20d"].shift(5)
    out["rv_ratio_5_20_map"] = out["rv_5d"] / out["rv_20d"]
    out["rv_ratio_20_60_map"] = out["rv_20d"] / out["rv_60d"]

    peak_120d = out["spy_adj_close"].rolling(120).max()
    out["drawdown_120d"] = out["spy_adj_close"] / peak_120d - 1.0
    return out


def fit_causal_stress_thresholds(
    rv_20d: pd.Series,
    *,
    min_periods: int,
    stress_quantile: float,
    high_stress_quantile: float,
) -> tuple[pd.Series, pd.Series]:
    """Return expanding historical RV thresholds using data through each row."""
    expanding = rv_20d.expanding(min_periods=min_periods)
    stress = expanding.quantile(stress_quantile)
    high = expanding.quantile(high_stress_quantile)
    return stress, high


def assign_broad_regimes(
    df: pd.DataFrame,
    config: BroadRegimeConfig | None = None,
) -> pd.DataFrame:
    """Assign one provisional broad regime to every sufficiently warm row.

    Rule hierarchy is deliberate:

    1. acute crisis: extreme stress plus ongoing fast deterioration;
    2. volatile rebound: high stress/damage plus positive short-term reversal;
    3. deterioration: stress/damage plus negative direction;
    4. easing/normalization: previously stressed/damaged state with falling RV;
    5. mild stress: elevated stress without a stronger directional morphology;
    6. calm: low stress and limited damage.

    Rows before the expanding stress thresholds warm up remain unlabeled.
    """
    cfg = config or BroadRegimeConfig()
    out = add_regime_features(df)

    stress_cut, high_cut = fit_causal_stress_thresholds(
        out["rv_20d"],
        min_periods=cfg.expanding_min_periods,
        stress_quantile=cfg.stress_quantile,
        high_stress_quantile=cfg.high_stress_quantile,
    )
    out["stress_cut"] = stress_cut
    out["high_stress_cut"] = high_cut

    valid = stress_cut.notna() & high_cut.notna()
    stressed = out["rv_20d"] >= stress_cut
    high_stress = out["rv_20d"] >= high_cut
    damaged = out["drawdown_120d"] <= cfg.damage_threshold
    severely_damaged = out["drawdown_120d"] <= cfg.severe_damage_threshold
    worsening = out["return_5d"] <= cfg.deterioration_5d_threshold
    rebounding = out["return_5d"] >= cfg.rebound_5d_threshold
    rv_rising_fast = (out["rv_5d_change_5d"] > 0.0) & (
        out["rv_ratio_5_20_map"] > 1.0
    )
    rv_falling = (out["rv_5d_change_5d"] < 0.0) & (
        out["rv_ratio_5_20_map"] < 1.0
    )

    regime = pd.Series(pd.NA, index=out.index, dtype="string")

    acute = valid & high_stress & (severely_damaged | worsening) & rv_rising_fast
    regime.loc[acute] = "acute_crisis"

    rebound = (
        valid
        & regime.isna()
        & (stressed | damaged)
        & rebounding
        & (rv_falling | (out["rv_20d_change_5d"] <= 0.0))
    )
    regime.loc[rebound] = "volatile_rebound"

    deterioration = (
        valid
        & regime.isna()
        & (stressed | damaged)
        & worsening
    )
    regime.loc[deterioration] = "deterioration"

    easing = (
        valid
        & regime.isna()
        & (stressed | damaged)
        & rv_falling
        & (out["return_5d"] > cfg.deterioration_5d_threshold)
    )
    regime.loc[easing] = "easing_normalization"

    mild = valid & regime.isna() & (stressed | damaged)
    regime.loc[mild] = "mild_stress"

    calm = valid & regime.isna()
    regime.loc[calm] = "calm"

    out["broad_regime"] = pd.Categorical(regime, categories=REGIME_ORDER)
    return out


def add_future_regime_target(
    df: pd.DataFrame,
    *,
    horizon: int = 20,
    regime_col: str = "broad_regime",
    target_col: str = "future_broad_regime",
) -> pd.DataFrame:
    """Add the regime observed ``horizon`` trading rows later."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if regime_col not in df.columns:
        raise KeyError(regime_col)

    out = df.copy()
    shifted = out[regime_col].astype("string").shift(-horizon)
    out[target_col] = pd.Categorical(shifted, categories=REGIME_ORDER)
    return out
