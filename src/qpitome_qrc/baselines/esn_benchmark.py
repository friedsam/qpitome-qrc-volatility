from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from qpitome_qrc.baselines.esn import (
    COMPACT_FEATURES,
    ESNConfig,
    EXPANDED_FEATURES,
    TARGET,
    fit_single_esn,
    grid_configs,
    summarize_run,
)
from qpitome_qrc.data.sequences import chronological_split, make_rolling_sequences
from qpitome_qrc.dimred.feature_select import CorrelationPruner
from qpitome_qrc.dimred.reducers import DimReductionConfig, build_dimred_pipeline


@dataclass(frozen=True)
class ESNFeatureSet:
    """Feature-set specification for ESN/QRC-style sequence benchmarks."""

    name: str
    features: list[str]
    transformer_builder: Callable[[], TransformerMixin] | None = None


@dataclass(frozen=True)
class ESNBenchmarkSpec:
    """One benchmark axis combination before ESN hyperparameters."""

    feature_set: str
    seq_len: int
    config: ESNConfig


@dataclass
class ESNBenchmarkResult:
    """Single ESN benchmark run plus metadata."""

    feature_set: str
    seq_len: int
    result: object


def hand_selected_features(columns: list[str]) -> list[str]:
    """Low-redundancy SPY+VIX feature set based on diagnostics."""
    candidates = [
        "spy_log_return",
        "spy_range",
        "spy_drawdown_20d",
        "rv_10d",
        "vix_close",
        "vix_pct_change",
    ]
    return [f for f in candidates if f in columns]


def default_esn_feature_sets(
    df: pd.DataFrame,
    pca_components: int = 6,
    corr_threshold: float = 0.95,
) -> list[ESNFeatureSet]:
    """Return feature sets for serious ESN benchmark tests.

    Transformers are fit on training rows only inside build_sequence_splits_for_feature_set.
    """
    columns = list(df.columns)
    compact = [f for f in COMPACT_FEATURES if f in columns]
    expanded = [f for f in EXPANDED_FEATURES if f in columns]
    hand = hand_selected_features(columns)

    feature_sets = [
        ESNFeatureSet(name="compact", features=compact),
        ESNFeatureSet(name="hand_selected", features=hand),
        ESNFeatureSet(
            name=f"compact_corr_pruned_{corr_threshold}",
            features=compact,
            transformer_builder=lambda: CorrelationPruner(threshold=corr_threshold, feature_names=compact),
        ),
        ESNFeatureSet(
            name=f"compact_pca_{pca_components}",
            features=compact,
            transformer_builder=lambda: build_dimred_pipeline(
                DimReductionConfig(kind="pca", n_components=pca_components, scale=True)
            ),
        ),
    ]

    if len(expanded) > len(compact):
        feature_sets.append(ESNFeatureSet(name="expanded", features=expanded))

    return feature_sets


def _split_tabular_masks(
    dates: pd.Series | np.ndarray,
    train_end: str = "2016-01-01",
    val_end: str = "2020-01-01",
) -> dict[str, np.ndarray]:
    date_values = pd.to_datetime(dates).to_numpy()
    train_end_dt = np.datetime64(train_end)
    val_end_dt = np.datetime64(val_end)
    return {
        "train": date_values < train_end_dt,
        "val": (date_values >= train_end_dt) & (date_values < val_end_dt),
        "test": date_values >= val_end_dt,
    }


def transform_feature_frame(
    df: pd.DataFrame,
    feature_set: ESNFeatureSet,
    train_end: str = "2016-01-01",
    val_end: str = "2020-01-01",
) -> tuple[pd.DataFrame, list[str]]:
    """Apply optional feature transformer fit on train rows only.

    Returns a DataFrame with date, target, and transformed feature columns.
    """
    missing = set(feature_set.features + ["date", TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {feature_set.name}: {sorted(missing)}")

    base = df[["date", TARGET, *feature_set.features]].copy()
    transformer = feature_set.transformer_builder() if feature_set.transformer_builder else None

    if transformer is None:
        return base, feature_set.features

    masks = _split_tabular_masks(base["date"], train_end=train_end, val_end=val_end)
    X_train = base.loc[masks["train"], feature_set.features]
    transformer.fit(X_train, base.loc[masks["train"], TARGET].to_numpy(dtype=int))

    X_all = transformer.transform(base[feature_set.features])
    if hasattr(transformer, "get_feature_names_out"):
        names = [str(x) for x in transformer.get_feature_names_out()]
    else:
        names = [f"{feature_set.name}_{i + 1}" for i in range(X_all.shape[1])]

    transformed = pd.DataFrame(X_all, columns=names, index=base.index)
    out = pd.concat([base[["date", TARGET]], transformed], axis=1)
    return out, names


def build_sequence_splits_for_feature_set(
    df: pd.DataFrame,
    feature_set: ESNFeatureSet,
    seq_len: int,
    train_end: str = "2016-01-01",
    val_end: str = "2020-01-01",
) -> tuple[dict, list[str]]:
    """Build chronological sequence splits for one feature set and window length."""
    transformed, feature_names = transform_feature_frame(
        df=df,
        feature_set=feature_set,
        train_end=train_end,
        val_end=val_end,
    )
    X, y, dates = make_rolling_sequences(
        df=transformed,
        features=feature_names,
        target=TARGET,
        seq_len=seq_len,
    )
    return chronological_split(X, y, dates, train_end=train_end, val_end=val_end), feature_names


def run_esn_benchmark_suite(
    df: pd.DataFrame,
    feature_sets: list[ESNFeatureSet],
    seq_lens: tuple[int, ...] = (20, 40),
    configs: list[ESNConfig] | None = None,
    train_end: str = "2016-01-01",
    val_end: str = "2020-01-01",
    include_test: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], ESNBenchmarkResult]]:
    """Run ESN benchmark across feature sets, sequence lengths, configs.

    The result key is (feature_set_name, seq_len, run_index). Seed aggregation can be
    done from the returned summary table using config columns.
    """
    if configs is None:
        configs = grid_configs(
            units=(300, 600),
            spectral_radius=(0.7, 0.9),
            leak_rate=(0.2, 0.5),
            reservoir_connectivity=(0.05, 0.1),
            readout_C=(0.1, 1.0),
            seeds=(1, 2, 3),
            washout=(0,),
            pooling=("final",),
            scale_states=(False,),
        )

    rows = []
    results: dict[tuple[str, int, int], ESNBenchmarkResult] = {}
    run_idx = 0

    for feature_set in feature_sets:
        for seq_len in seq_lens:
            splits, feature_names = build_sequence_splits_for_feature_set(
                df=df,
                feature_set=feature_set,
                seq_len=seq_len,
                train_end=train_end,
                val_end=val_end,
            )
            for config in configs:
                run_idx += 1
                print(f"[{run_idx}] {feature_set.name} | seq_len={seq_len} | {config}")
                result = fit_single_esn(splits=splits, config=config, tune_threshold=True)
                row = summarize_run(result, include_test=include_test)
                row.update(
                    {
                        "feature_set": feature_set.name,
                        "seq_len": seq_len,
                        "n_input_features": len(feature_names),
                        "feature_names": feature_names,
                        "run_idx": run_idx,
                    }
                )
                rows.append(row)
                results[(feature_set.name, seq_len, run_idx)] = ESNBenchmarkResult(
                    feature_set=feature_set.name,
                    seq_len=seq_len,
                    result=result,
                )

    summary = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    return summary, results


def aggregate_esn_seeds(summary: pd.DataFrame, metric: str = "val_pr_auc") -> pd.DataFrame:
    """Aggregate ESN benchmark rows across seeds for stable model selection."""
    config_cols = [
        "feature_set",
        "seq_len",
        "n_input_features",
        "units",
        "spectral_radius",
        "leak_rate",
        "input_scaling",
        "input_connectivity",
        "reservoir_connectivity",
        "readout_C",
        "washout",
        "pooling",
        "scale_states",
    ]
    config_cols = [c for c in config_cols if c in summary.columns]

    agg = (
        summary.groupby(config_cols, as_index=False)
        .agg(
            mean_val_pr_auc=("val_pr_auc", "mean"),
            std_val_pr_auc=("val_pr_auc", "std"),
            mean_val_f1=("val_f1_class_1", "mean"),
            mean_test_pr_auc=("test_pr_auc", "mean") if "test_pr_auc" in summary else (metric, "mean"),
            n_seeds=(metric, "size"),
        )
        .sort_values("mean_val_pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def save_esn_benchmark_outputs(
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    output_dir: str | Path = "reports/esn_benchmark/tables",
) -> Path:
    """Save ESN benchmark summary tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "esn_benchmark_runs.csv", index=False)
    aggregate.to_csv(output_dir / "esn_benchmark_seed_aggregate.csv", index=False)
    return output_dir
