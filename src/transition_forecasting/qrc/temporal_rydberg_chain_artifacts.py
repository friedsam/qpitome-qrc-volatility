from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_METADATA_COLUMNS = (
    "fold",
    "sample_id",
    "fold_split",
    "lead",
    "label",
    "episode_id",
    "origin_date",
)


def signal_diagnostics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Return simple amplitude, bias, and association diagnostics."""
    observed = np.asarray(y, dtype=float).reshape(-1)
    forecast = np.asarray(prediction, dtype=float).reshape(-1)
    finite = np.isfinite(observed) & np.isfinite(forecast)
    observed = observed[finite]
    forecast = forecast[finite]
    if observed.size == 0:
        return {
            "target_std": np.nan,
            "prediction_std": np.nan,
            "mean_error": np.nan,
            "correlation": np.nan,
        }

    target_std = float(np.std(observed))
    prediction_std = float(np.std(forecast))
    if target_std <= 0.0 or prediction_std <= 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(observed, forecast)[0, 1])
    return {
        "target_std": target_std,
        "prediction_std": prediction_std,
        "mean_error": float(np.mean(forecast - observed)),
        "correlation": correlation,
    }


def feature_names_from_metadata(
    metadata: dict[str, object],
) -> tuple[str, ...]:
    """Reconstruct the exact feature order emitted by the reservoir."""
    probe_steps = [int(value) for value in metadata["probe_steps"]]
    nearest_pairs = [
        (int(pair[0]), int(pair[1]))
        for pair in metadata["nearest_pairs"]
    ]
    long_pairs = [
        (int(pair[0]), int(pair[1]))
        for pair in metadata["long_pairs"]
    ]
    positions = np.asarray(metadata["positions_um"], dtype=float)
    n_atoms = int(positions.shape[0])

    base_names: list[str] = []
    base_names.extend(f"occupation_site_{site}" for site in range(n_atoms))
    base_names.extend(
        f"nearest_pair_{left}_{right}" for left, right in nearest_pairs
    )
    base_names.extend(
        f"nearest_connected_{left}_{right}" for left, right in nearest_pairs
    )
    base_names.extend(("excitation_density", "mean_domain_wall"))
    base_names.extend(
        f"long_pair_{left}_{right}" for left, right in long_pairs
    )
    base_names.extend(
        f"long_connected_{left}_{right}" for left, right in long_pairs
    )

    names = tuple(
        f"probe_{probe}_{name}"
        for probe in probe_steps
        for name in base_names
    )
    expected = int(metadata["feature_count"])
    if len(names) != expected:
        raise ValueError(
            "feature-name reconstruction does not match emitted feature count: "
            f"names={len(names)}, emitted={expected}"
        )
    return names


def make_prediction_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    fold: int,
    condition: str,
    alpha: float | None,
    readout_mode: str,
) -> pd.DataFrame:
    """Return one row per sample and forecast horizon."""
    observed = np.asarray(y, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    if observed.shape != forecast.shape or observed.ndim != 2:
        raise ValueError(
            "y and prediction must be aligned two-dimensional path arrays"
        )
    if len(frame) != len(observed):
        raise ValueError("frame and prediction arrays are not aligned")

    horizons = observed.shape[1]
    payload: dict[str, object] = {
        column: np.repeat(frame[column].to_numpy(), horizons)
        for column in _METADATA_COLUMNS
        if column in frame.columns
    }
    payload.update(
        {
            "fold": np.repeat(int(fold), len(frame) * horizons),
            "condition": np.repeat(condition, len(frame) * horizons),
            "readout_mode": np.repeat(readout_mode, len(frame) * horizons),
            "alpha": np.repeat(
                np.nan if alpha is None else float(alpha),
                len(frame) * horizons,
            ),
            "horizon": np.tile(np.arange(1, horizons + 1), len(frame)),
            "y_true": observed.reshape(-1),
            "y_pred": forecast.reshape(-1),
        }
    )
    return pd.DataFrame(payload)


def best_alpha_per_condition(metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose one development alpha per condition by mean fold QLIKE."""
    candidates = metrics.loc[
        ~metrics["condition"].eq("har_only")
        & metrics["alpha"].notna()
    ].copy()
    grouped = (
        candidates.groupby(["condition", "alpha"], as_index=False)
        .agg(
            mean_val_qlike=("val_qlike", "mean"),
            mean_val_rmse=("val_rmse", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(
            ["condition", "mean_val_qlike", "mean_val_rmse", "alpha"]
        )
    )
    return grouped.groupby("condition", as_index=False).head(1).reset_index(
        drop=True
    )


def write_feature_archive(
    path: Path,
    *,
    feature_blocks: Iterable[dict[str, object]],
    feature_names: tuple[str, ...],
) -> None:
    """Persist readout-ready reservoir features without object arrays."""
    blocks = list(feature_blocks)
    if not blocks:
        raise ValueError("feature_blocks cannot be empty")

    features = np.concatenate(
        [np.asarray(block["features"], dtype=float) for block in blocks],
        axis=0,
    )
    targets = np.concatenate(
        [np.asarray(block["targets"], dtype=float) for block in blocks],
        axis=0,
    )
    har_predictions = np.concatenate(
        [np.asarray(block["har_predictions"], dtype=float) for block in blocks],
        axis=0,
    )
    source_level = np.concatenate(
        [np.asarray(block["source_level"], dtype=float) for block in blocks],
        axis=0,
    )
    encoded_level = np.concatenate(
        [np.asarray(block["encoded_level"], dtype=float) for block in blocks],
        axis=0,
    )
    if features.shape[1] != len(feature_names):
        raise ValueError(
            "feature matrix width and feature_names length do not match"
        )

    metadata: dict[str, np.ndarray] = {}
    for column in _METADATA_COLUMNS:
        values = []
        for block in blocks:
            frame = block["frame"]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("feature block frame must be a DataFrame")
            values.append(frame[column].to_numpy())
        combined = np.concatenate(values)
        metadata[column] = (
            combined.astype(int)
            if column in {"fold", "lead", "label"}
            else combined.astype(str)
        )
    metadata["condition"] = np.concatenate(
        [
            np.repeat(str(block["condition"]), len(block["frame"]))
            for block in blocks
        ]
    ).astype(str)

    np.savez_compressed(
        path,
        feature_matrix=features,
        feature_names=np.asarray(feature_names, dtype=str),
        target_path=targets,
        har_prediction_path=har_predictions,
        source_level_windows=source_level,
        encoded_level_windows=encoded_level,
        **metadata,
    )


def _prediction_path(
    predictions: pd.DataFrame,
    *,
    sample_key: tuple[int, str] | None,
    lead: int,
    condition: str,
    alpha: float | None,
) -> pd.Series:
    rows = predictions.loc[
        predictions["fold_split"].eq("val")
        & predictions["label"].eq(1)
        & predictions["lead"].eq(lead)
        & predictions["condition"].eq(condition)
    ].copy()
    if sample_key is not None:
        sample_fold, sample_id = sample_key
        rows = rows.loc[
            rows["fold"].eq(int(sample_fold))
            & rows["sample_id"].astype(str).eq(str(sample_id))
        ]
    if alpha is not None:
        rows = rows.loc[np.isclose(rows["alpha"], float(alpha))]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby("horizon", sort=True)["y_pred"].mean()


def _actual_path(
    predictions: pd.DataFrame,
    *,
    sample_key: tuple[int, str] | None,
    lead: int,
) -> pd.Series:
    rows = predictions.loc[
        predictions["fold_split"].eq("val")
        & predictions["label"].eq(1)
        & predictions["lead"].eq(lead)
        & predictions["condition"].eq("har_only")
    ].copy()
    if sample_key is not None:
        sample_fold, sample_id = sample_key
        rows = rows.loc[
            rows["fold"].eq(int(sample_fold))
            & rows["sample_id"].astype(str).eq(str(sample_id))
        ]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby("horizon", sort=True)["y_true"].mean()


def _plot_overlay(
    *,
    path: Path,
    history: np.ndarray,
    actual: pd.Series,
    predictions: pd.DataFrame,
    sample_key: tuple[int, str] | None,
    lead: int,
    alpha_map: dict[str, float],
    conditions: tuple[str, ...],
    title: str,
) -> None:
    x_history = np.arange(-len(history) + 1, 1)
    x_future = actual.index.to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(x_history, history, label="Observed history")
    ax.plot(x_future, actual.to_numpy(dtype=float), label="Actual future")
    for condition in conditions:
        alpha = None if condition == "har_only" else alpha_map.get(condition)
        if condition != "har_only" and alpha is None:
            continue
        path_values = _prediction_path(
            predictions,
            sample_key=sample_key,
            lead=lead,
            condition=condition,
            alpha=alpha,
        )
        if path_values.empty:
            continue
        label = "HAR" if condition == "har_only" else condition
        if alpha is not None:
            label = f"{label} (alpha={alpha:g})"
        ax.plot(
            path_values.index.to_numpy(dtype=int),
            path_values.to_numpy(dtype=float),
            label=label,
        )

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.axvline(float(lead), linestyle=":", linewidth=1.0)
    ax.set_xlabel("Trading days relative to forecast origin")
    ax.set_ylabel("Log volatility")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_transition_overlays(
    run_dir: Path,
    *,
    predictions: pd.DataFrame,
    window_records: pd.DataFrame,
    best_alphas: pd.DataFrame,
) -> list[str]:
    """Write mean and peak-event overlays for each transition lead."""
    alpha_map = {
        str(row.condition): float(row.alpha)
        for row in best_alphas.itertuples(index=False)
    }
    files: list[str] = []
    validation = window_records.loc[
        window_records["fold_split"].eq("val")
        & window_records["label"].eq(1)
    ].copy()

    for lead in sorted(validation["lead"].unique()):
        lead = int(lead)
        lead_windows = validation.loc[validation["lead"].eq(lead)]
        if lead_windows.empty:
            continue
        history = np.mean(
            np.stack(lead_windows["level_window"].to_list()),
            axis=0,
        )
        actual = _actual_path(predictions, sample_key=None, lead=lead)
        if actual.empty:
            continue

        mean_name = f"transition_mean_overlay_lead_{lead}.png"
        _plot_overlay(
            path=run_dir / mean_name,
            history=history,
            actual=actual,
            predictions=predictions,
            sample_key=None,
            lead=lead,
            alpha_map=alpha_map,
            conditions=(
                "har_only",
                "ordered",
                "shuffled",
                "reset",
                "rate_only",
            ),
            title=f"Mean transition forecast anatomy: lead {lead}",
        )
        files.append(mean_name)

        actual_rows = predictions.loc[
            predictions["fold_split"].eq("val")
            & predictions["label"].eq(1)
            & predictions["lead"].eq(lead)
            & predictions["condition"].eq("har_only")
        ]
        severity = actual_rows.groupby(["fold", "sample_id"])["y_true"].max()
        if severity.empty:
            continue
        sample_fold, sample_id = severity.idxmax()
        sample_key = (int(sample_fold), str(sample_id))
        sample_window = lead_windows.loc[
            lead_windows["fold"].eq(int(sample_fold))
            & lead_windows["sample_id"].astype(str).eq(str(sample_id))
        ]
        if sample_window.empty:
            continue
        peak_actual = _actual_path(
            predictions,
            sample_key=sample_key,
            lead=lead,
        )
        peak_name = f"transition_peak_event_overlay_lead_{lead}.png"
        _plot_overlay(
            path=run_dir / peak_name,
            history=np.asarray(
                sample_window.iloc[0]["level_window"], dtype=float
            ),
            actual=peak_actual,
            predictions=predictions,
            sample_key=sample_key,
            lead=lead,
            alpha_map=alpha_map,
            conditions=("har_only", "ordered", "reset", "rate_only"),
            title=(
                f"Strongest validation transition: lead {lead}, "
                f"fold {sample_fold}, sample {sample_id}"
            ),
        )
        files.append(peak_name)
    return files
