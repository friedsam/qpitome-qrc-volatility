from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.fold_selection import (
    DEVELOPMENT_FOLD_SPLITS,
    select_balanced_episode_rows,
    validate_fold_manifest_schema,
)
from transition_forecasting.modeling.stage_e_classical_baselines import (
    TARGET_COLUMNS,
)
from transition_forecasting.modeling.stage_e_fixed_spec_diagnostics import (
    horizon_mz_summary,
    mincer_zarnowitz,
)
from transition_forecasting.modeling.stage_e_sequence_models import (
    har_predictions,
    metrics,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    SUPPORTED_CONDITIONS,
    Condition,
    TemporalRydbergChainConfig,
    build_temporal_rydberg_chain_features,
    controlled_level_windows,
    effective_rank,
    fit_level_rate_scaler,
    transform_level_windows,
)


@dataclass(frozen=True)
class TemporalRydbergExperimentConfig:
    folds: tuple[int, ...] = (1, 2, 3)
    leads: tuple[int, ...] = (1, 5, 10)
    conditions: tuple[Condition, ...] = SUPPORTED_CONDITIONS
    sequence_length: int = 40
    max_per_class: int = 24
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    readout_mode: str = "direct"
    seed: int = 20260721
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds:
            raise ValueError("folds cannot be empty")
        if not self.leads:
            raise ValueError("leads cannot be empty")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must be positive")
        if self.readout_mode not in {"direct", "har_residual"}:
            raise ValueError(
                "readout_mode must be direct or har_residual"
            )
        unknown = set(self.conditions).difference(SUPPORTED_CONDITIONS)
        if unknown:
            raise ValueError(
                f"unsupported conditions: {sorted(unknown)}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FoldTensor:
    values: np.ndarray
    sample_ids: np.ndarray
    valid: np.ndarray
    channel_names: tuple[str, ...]


def _load_excluded_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{path} has no sample_id column")
    return set(frame["sample_id"].astype(str))


def load_fold_tensor(path: Path) -> FoldTensor:
    with np.load(path, allow_pickle=True) as bundle:
        required = {"X", "sample_id"}
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(
                f"{path} missing arrays: {sorted(missing)}"
            )
        values = np.asarray(bundle["X"], dtype=float)
        sample_ids = np.asarray(
            bundle["sample_id"],
            dtype=object,
        ).astype(str)
        valid = (
            np.asarray(bundle["valid"], dtype=bool)
            if "valid" in bundle.files
            else np.ones(len(values), dtype=bool)
        )
        channel_names = (
            tuple(
                str(value)
                for value in np.asarray(bundle["channel_names"])
            )
            if "channel_names" in bundle.files
            else tuple()
        )
    if values.ndim not in (2, 3):
        raise ValueError(
            f"{path}: X must be two- or three-dimensional, got {values.shape}"
        )
    if len(values) != len(sample_ids) or len(values) != len(valid):
        raise ValueError(
            f"{path}: tensor metadata lengths are inconsistent"
        )
    return FoldTensor(
        values=values,
        sample_ids=sample_ids,
        valid=valid,
        channel_names=channel_names,
    )


def resolve_level_channel(
    tensor: FoldTensor,
    *,
    name: str,
    fallback: int,
) -> int | None:
    if tensor.values.ndim == 2:
        return None
    if name in tensor.channel_names:
        return tensor.channel_names.index(name)
    if fallback < 0 or fallback >= tensor.values.shape[2]:
        raise ValueError(
            f"fallback level channel {fallback} outside tensor shape "
            f"{tensor.values.shape}"
        )
    return int(fallback)


def extract_level_windows(
    tensor: FoldTensor,
    row_indices: np.ndarray,
    *,
    sequence_length: int,
    level_channel: int | None,
) -> np.ndarray:
    if tensor.values.shape[1] < sequence_length:
        raise ValueError(
            f"requested {sequence_length} steps but tensor has "
            f"{tensor.values.shape[1]}"
        )
    selected = tensor.values[
        row_indices,
        -sequence_length:,
    ]
    if selected.ndim == 3:
        if level_channel is None:
            raise ValueError(
                "a channel index is required for a three-dimensional tensor"
            )
        selected = selected[:, :, level_channel]
    level = np.asarray(selected, dtype=float)
    if level.ndim != 2:
        raise ValueError(
            f"level extraction produced unexpected shape {level.shape}"
        )
    return level


def _condition_seed(base_seed: int, fold: int, condition: str) -> int:
    offset = sum(
        (index + 1) * ord(character)
        for index, character in enumerate(condition)
    )
    return int(base_seed + 1_000_003 * fold + offset)


def _metric_row(
    *,
    fold: int,
    condition: str,
    alpha: float | None,
    readout_mode: str,
    y: np.ndarray,
    prediction: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, object]:
    qlike, rmse = metrics(y, prediction, val_mask)
    val_y = y[val_mask]
    val_prediction = prediction[val_mask]
    return {
        "fold": int(fold),
        "condition": condition,
        "readout_mode": readout_mode,
        "alpha": np.nan if alpha is None else float(alpha),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "val_qlike": float(qlike),
        "val_rmse": float(rmse),
        **mincer_zarnowitz(val_y, val_prediction),
        **horizon_mz_summary(val_y, val_prediction),
    }


def _group_metric_rows(
    *,
    fold: int,
    condition: str,
    alpha: float | None,
    readout_mode: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    val_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    validation = frame.loc[val_mask].reset_index(drop=True)
    y_val = y[val_mask]
    prediction_val = prediction[val_mask]
    for lead in sorted(validation["lead"].dropna().unique()):
        for label in sorted(validation["label"].dropna().unique()):
            mask = (
                validation["lead"].eq(lead).to_numpy()
                & validation["label"].eq(label).to_numpy()
            )
            if not mask.any():
                continue
            local_mask = np.ones(int(mask.sum()), dtype=bool)
            qlike, rmse = metrics(
                y_val[mask],
                prediction_val[mask],
                local_mask,
            )
            rows.append(
                {
                    "fold": int(fold),
                    "condition": condition,
                    "readout_mode": readout_mode,
                    "alpha": np.nan if alpha is None else float(alpha),
                    "lead": int(lead),
                    "label": int(label),
                    "samples": int(mask.sum()),
                    "qlike": float(qlike),
                    "rmse": float(rmse),
                }
            )
    return rows


def _feature_diagnostic_row(
    *,
    fold: int,
    condition: str,
    features: np.ndarray,
    train_mask: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, object]:
    train = np.asarray(features[train_mask], dtype=float)
    std = train.std(axis=0)
    centered = train - train.mean(axis=0, keepdims=True)
    return {
        "fold": int(fold),
        "condition": condition,
        "samples": int(len(features)),
        "features": int(features.shape[1]),
        "effective_rank_train": effective_rank(train),
        "numerical_rank_train": int(np.linalg.matrix_rank(centered)),
        "near_constant_features": int((std < 1e-8).sum()),
        "mean_feature_std": float(std.mean()),
        "max_feature_std": float(std.max(initial=0.0)),
        "probe_steps": json.dumps(metadata["probe_steps"]),
        "total_evolution_time_us": float(
            metadata["total_evolution_time_us"]
        ),
    }


def _select_rows_for_fold(
    manifest: pd.DataFrame,
    *,
    fold: int,
    leads: tuple[int, ...],
    max_per_class: int,
    excluded_ids: set[str],
    seed: int,
) -> pd.DataFrame:
    parts = [
        select_balanced_episode_rows(
            manifest,
            fold=fold,
            lead=int(lead),
            max_per_class=max_per_class,
            excluded_sample_ids=excluded_ids,
            splits=DEVELOPMENT_FOLD_SPLITS,
            seed=seed + int(lead) * 1009,
        )
        for lead in leads
    ]
    selected = pd.concat(parts, ignore_index=True)
    if selected["sample_id"].astype(str).duplicated().any():
        raise RuntimeError(
            f"fold {fold}: duplicate sample IDs across selected leads"
        )
    if selected["fold_split"].eq("test").any():
        raise RuntimeError(
            "development selection unexpectedly included test rows"
        )
    return selected.sort_values(
        [
            "fold_split",
            "lead",
            "label",
            "episode_id",
            "origin_date",
            "sample_id",
        ]
    ).reset_index(drop=True)


def run_temporal_rydberg_chain_experiment(
    *,
    manifest_path: Path,
    tensor_root: Path,
    results_root: Path,
    experiment: TemporalRydbergExperimentConfig,
    reservoir: TemporalRydbergChainConfig,
    contaminated_samples: Path | None = None,
    run_id: str | None = None,
) -> Path:
    experiment.validate()
    reservoir.validate()
    manifest = pd.read_csv(manifest_path)
    validate_fold_manifest_schema(manifest)
    excluded_ids = _load_excluded_ids(contaminated_samples)

    parameters = {
        "manifest_path": manifest_path,
        "tensor_root": tensor_root,
        "contaminated_samples": contaminated_samples,
        "experiment": experiment.to_dict(),
        "reservoir": reservoir.to_dict(),
    }
    run_dir = begin_run(
        results_root,
        parameters,
        run_id=run_id,
    )

    metric_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    retained_rows: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, object]] = []

    for fold in experiment.folds:
        selected = _select_rows_for_fold(
            manifest,
            fold=int(fold),
            leads=experiment.leads,
            max_per_class=experiment.max_per_class,
            excluded_ids=excluded_ids,
            seed=experiment.seed,
        )
        tensor_path = (
            tensor_root
            / f"fold_{fold}"
            / "compact_tensor.npz"
        )
        tensor = load_fold_tensor(tensor_path)
        row_by_id = {
            sample_id: index
            for index, sample_id in enumerate(tensor.sample_ids)
        }
        tensor_rows = np.asarray(
            [
                row_by_id.get(str(sample_id), -1)
                for sample_id in selected["sample_id"]
            ],
            dtype=int,
        )
        if np.any(tensor_rows < 0):
            missing = selected.loc[
                tensor_rows < 0,
                "sample_id",
            ].astype(str).tolist()
            raise RuntimeError(
                f"fold {fold}: selected samples absent from tensor: "
                f"{missing[:5]}"
            )

        level_channel = resolve_level_channel(
            tensor,
            name=experiment.level_channel_name,
            fallback=experiment.fallback_level_channel,
        )
        level = extract_level_windows(
            tensor,
            tensor_rows,
            sequence_length=experiment.sequence_length,
            level_channel=level_channel,
        )
        usable = (
            tensor.valid[tensor_rows]
            & np.isfinite(level).all(axis=1)
        )
        frame = selected.loc[usable].reset_index(drop=True)
        level = level[usable]
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no valid rows remain")
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        if not train_mask.any() or not val_mask.any():
            raise RuntimeError(
                f"fold {fold}: no train or validation rows remain"
            )

        scaler = fit_level_rate_scaler(level, train_mask)
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        metric_rows.append(
            _metric_row(
                fold=int(fold),
                condition="har_only",
                alpha=None,
                readout_mode="baseline",
                y=y,
                prediction=har,
                train_mask=train_mask,
                val_mask=val_mask,
            )
        )
        group_rows.extend(
            _group_metric_rows(
                fold=int(fold),
                condition="har_only",
                alpha=None,
                readout_mode="baseline",
                frame=frame,
                y=y,
                prediction=har,
                val_mask=val_mask,
            )
        )

        condition_features: dict[str, np.ndarray] = {}
        condition_metadata: dict[str, dict[str, object]] = {}
        for condition in experiment.conditions:
            controlled = controlled_level_windows(
                level,
                condition,
                seed=_condition_seed(
                    experiment.seed,
                    int(fold),
                    condition,
                ),
            )
            windows = transform_level_windows(controlled, scaler)
            features, metadata = (
                build_temporal_rydberg_chain_features(
                    windows,
                    reservoir,
                    condition=condition,
                )
            )
            condition_features[condition] = features
            condition_metadata[condition] = metadata
            diagnostic_rows.append(
                _feature_diagnostic_row(
                    fold=int(fold),
                    condition=condition,
                    features=features,
                    train_mask=train_mask,
                    metadata=metadata,
                )
            )

            feature_scaler = StandardScaler()
            feature_train = feature_scaler.fit_transform(
                features[train_mask]
            )
            feature_all = feature_scaler.transform(features)
            target = (
                y
                if experiment.readout_mode == "direct"
                else y - har
            )
            for alpha in experiment.alphas:
                readout = Ridge(alpha=float(alpha))
                readout.fit(
                    feature_train,
                    target[train_mask],
                )
                correction = readout.predict(feature_all)
                prediction = (
                    correction
                    if experiment.readout_mode == "direct"
                    else har + correction
                )
                metric_rows.append(
                    _metric_row(
                        fold=int(fold),
                        condition=condition,
                        alpha=float(alpha),
                        readout_mode=experiment.readout_mode,
                        y=y,
                        prediction=prediction,
                        train_mask=train_mask,
                        val_mask=val_mask,
                    )
                )
                group_rows.extend(
                    _group_metric_rows(
                        fold=int(fold),
                        condition=condition,
                        alpha=float(alpha),
                        readout_mode=experiment.readout_mode,
                        frame=frame,
                        y=y,
                        prediction=prediction,
                        val_mask=val_mask,
                    )
                )

        ordered = condition_features.get("ordered")
        if ordered is not None:
            for condition, features in condition_features.items():
                if condition == "ordered":
                    continue
                diagnostic_rows.append(
                    {
                        "fold": int(fold),
                        "condition": f"ordered_minus_{condition}",
                        "samples": int(len(features)),
                        "features": int(features.shape[1]),
                        "effective_rank_train": np.nan,
                        "numerical_rank_train": np.nan,
                        "near_constant_features": np.nan,
                        "mean_feature_std": np.nan,
                        "max_feature_std": np.nan,
                        "probe_steps": json.dumps(
                            condition_metadata[condition]["probe_steps"]
                        ),
                        "total_evolution_time_us": float(
                            condition_metadata[condition][
                                "total_evolution_time_us"
                            ]
                        ),
                        "mean_absolute_feature_difference": float(
                            np.mean(np.abs(ordered - features))
                        ),
                    }
                )

        retained = frame[
            [
                "sample_id",
                "fold",
                "fold_split",
                "lead",
                "label",
                "episode_id",
                "origin_date",
            ]
        ].copy()
        retained["tensor_row"] = tensor_rows[usable]
        retained_rows.append(retained)
        fold_metadata.append(
            {
                "fold": int(fold),
                "tensor_path": str(tensor_path),
                "tensor_shape": list(tensor.values.shape),
                "channel_names": list(tensor.channel_names),
                "level_channel": level_channel,
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "scaler": scaler.to_dict(),
                "reservoir_metadata": condition_metadata.get(
                    "ordered"
                ),
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    group_frame = pd.DataFrame(group_rows)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    retained_frame = pd.concat(retained_rows, ignore_index=True)
    metrics_frame.to_csv(
        run_dir / "fold_metrics.csv",
        index=False,
    )
    group_frame.to_csv(
        run_dir / "group_metrics.csv",
        index=False,
    )
    diagnostics_frame.to_csv(
        run_dir / "feature_diagnostics.csv",
        index=False,
    )
    retained_frame.to_csv(
        run_dir / "retained_samples.csv",
        index=False,
    )

    candidates = metrics_frame.loc[
        ~metrics_frame["condition"].eq("har_only")
    ]
    best_per_fold = (
        candidates.sort_values(
            [
                "fold",
                "val_qlike",
                "val_rmse",
                "condition",
                "alpha",
            ]
        )
        .groupby("fold", as_index=False)
        .head(1)
    )
    best_per_fold.to_csv(
        run_dir / "best_per_fold.csv",
        index=False,
    )
    summary = {
        "schema_version": 1,
        "status": "development_assay",
        "test_rows_used": 0,
        "manifest_path": str(manifest_path),
        "tensor_root": str(tensor_root),
        "experiment": experiment.to_dict(),
        "reservoir": reservoir.to_dict(),
        "folds": fold_metadata,
        "files": {
            "fold_metrics": "fold_metrics.csv",
            "group_metrics": "group_metrics.csv",
            "feature_diagnostics": "feature_diagnostics.csv",
            "retained_samples": "retained_samples.csv",
            "best_per_fold": "best_per_fold.csv",
        },
        "known_limitations": [
            "Development folds only; no untouched test rows are evaluated.",
            "Exact-state simulation is intended for the initial small-atom "
            "architecture assay.",
            "Validation metrics are used to inspect the alpha grid and are "
            "not final unbiased estimates.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return run_dir
