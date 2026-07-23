from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    TARGET_COLUMNS,
)
from transition_forecasting.modeling.stage_e_sequence_models import metrics
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    cache_identity,
    evolve_ladder_probe_probabilities,
    load_probability_cache,
    payload_sha256,
    probabilities_to_symmetric_modes,
    shot_modes_from_probabilities,
    write_probability_cache,
)
from transition_forecasting.qrc.ladder_readout_upgrade_tools import (
    LadderReadoutUpgradeConfig,
    apply_calibration,
    select_inner_configuration,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    build_candidate_sequences,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _metric_payload,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _resolve_probe_steps,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


ShotProtocol = Literal["shot_consistent", "exact_train_shot_val"]
SHOT_PROTOCOLS: tuple[ShotProtocol, ...] = (
    "shot_consistent",
    "exact_train_shot_val",
)
MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("ladder_linear_segmented", "linear"),
    ("ladder_poly2_segmented", "polynomial_degree2"),
)


@dataclass(frozen=True)
class LadderFiniteShotStudyConfig:
    """Finite-shot robustness protocol for the two frozen segmented ladder heads."""

    folds: tuple[int, ...] = tuple(range(1, 9))
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    sequence_length: int = 40
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    split_horizon: int = 4
    shot_counts: tuple[int, ...] = (250, 500, 1000, 2000, 5000)
    shot_seeds: tuple[int, ...] = (
        20260731,
        20260732,
        20260733,
        20260734,
        20260735,
    )
    protocols: tuple[ShotProtocol, ...] = SHOT_PROTOCOLS
    global_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    early_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5)
    transition_lambdas: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
    )
    linear_alpha: float = 100.0
    polynomial_alphas: tuple[float, ...] = (100.0, 1000.0, 10000.0)
    polynomial_degree: int = 2
    ladder_interaction_scale: float = 1.25
    seed: int = 20260722
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or any(int(value) < 1 for value in self.folds):
            raise ValueError("folds must be positive and nonempty")
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be unique")
        if not self.leads or any(int(value) < 1 for value in self.leads):
            raise ValueError("leads must be positive and nonempty")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not 1 <= self.split_horizon < 10:
            raise ValueError("split_horizon must lie between 1 and 9")
        if not self.shot_counts or any(int(value) < 1 for value in self.shot_counts):
            raise ValueError("shot_counts must be positive and nonempty")
        if len(set(self.shot_counts)) != len(self.shot_counts):
            raise ValueError("shot_counts must be unique")
        if not self.shot_seeds or len(set(self.shot_seeds)) != len(self.shot_seeds):
            raise ValueError("shot_seeds must be nonempty and unique")
        if not self.protocols or set(self.protocols).difference(SHOT_PROTOCOLS):
            raise ValueError("protocols contain an unsupported finite-shot protocol")
        if self.linear_alpha <= 0:
            raise ValueError("linear_alpha must be positive")
        if not self.polynomial_alphas or any(
            float(value) <= 0 for value in self.polynomial_alphas
        ):
            raise ValueError("polynomial_alphas must be positive")
        if self.polynomial_degree != 2:
            raise ValueError("polynomial_degree must remain two")
        if self.ladder_interaction_scale <= 0:
            raise ValueError("ladder_interaction_scale must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def readout_config(self) -> LadderReadoutUpgradeConfig:
        return LadderReadoutUpgradeConfig(
            folds=self.folds,
            leads=self.leads,
            max_per_class=self.max_per_class,
            sequence_length=self.sequence_length,
            prequential_blocks=self.prequential_blocks,
            inner_holdout_fraction=self.inner_holdout_fraction,
            split_horizon=self.split_horizon,
            global_lambdas=self.global_lambdas,
            early_lambdas=self.early_lambdas,
            transition_lambdas=self.transition_lambdas,
            linear_alpha=self.linear_alpha,
            polynomial_alphas=self.polynomial_alphas,
            polynomial_degree=self.polynomial_degree,
            ladder_interaction_scale=self.ladder_interaction_scale,
            seed=self.seed,
            level_channel_name=self.level_channel_name,
            fallback_level_channel=self.fallback_level_channel,
        )


def protocol_feature_matrix(
    *,
    exact_modes: np.ndarray,
    sampled_modes: np.ndarray,
    train_mask: np.ndarray,
    protocol: ShotProtocol,
) -> np.ndarray:
    exact = np.asarray(exact_modes, dtype=float)
    sampled = np.asarray(sampled_modes, dtype=float)
    train = np.asarray(train_mask, dtype=bool)
    if exact.shape != sampled.shape or exact.ndim != 2:
        raise ValueError("exact and sampled mode matrices must be aligned")
    if train.shape != (len(exact),):
        raise ValueError("train_mask must align with mode matrices")
    if protocol == "shot_consistent":
        return sampled.copy()
    if protocol == "exact_train_shot_val":
        hybrid = sampled.copy()
        hybrid[train] = exact[train]
        return hybrid
    raise ValueError(f"unsupported finite-shot protocol: {protocol}")


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = pd.Series(np.asarray(left, dtype=float).reshape(-1)).rank(method="average")
    y = pd.Series(np.asarray(right, dtype=float).reshape(-1)).rank(method="average")
    if x.std() <= 0.0 or y.std() <= 0.0:
        return float("nan")
    return float(x.corr(y))


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if x.size == 0 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _prediction_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    har: np.ndarray,
    validation: np.ndarray,
    *,
    model_name: str,
    protocol: str,
    shots: int,
    shot_seed: int,
    ridge_alpha: float,
    early_lambda: float,
    transition_lambda: float,
) -> pd.DataFrame:
    selected = frame.loc[validation].reset_index(drop=True)
    horizons = int(y.shape[1])
    count = int(validation.sum()) * horizons
    return pd.DataFrame(
        {
            "fold": np.repeat(selected["fold"].to_numpy(dtype=int), horizons),
            "sample_id": np.repeat(
                selected["sample_id"].astype(str).to_numpy(), horizons
            ),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "episode_id": np.repeat(
                selected["episode_id"].astype(str).to_numpy(), horizons
            ),
            "origin_date": np.repeat(
                selected["origin_date"].astype(str).to_numpy(), horizons
            ),
            "evaluation_split": np.repeat("val", count),
            "model_name": np.repeat(model_name, count),
            "protocol": np.repeat(protocol, count),
            "shots": np.repeat(int(shots), count),
            "shot_seed": np.repeat(int(shot_seed), count),
            "ridge_alpha": np.repeat(float(ridge_alpha), count),
            "early_lambda": np.repeat(float(early_lambda), count),
            "transition_lambda": np.repeat(float(transition_lambda), count),
            "horizon": np.tile(np.arange(1, horizons + 1), int(validation.sum())),
            "y_true": np.asarray(y)[validation].reshape(-1),
            "y_pred": np.asarray(prediction)[validation].reshape(-1),
            "har_pred": np.asarray(har)[validation].reshape(-1),
        }
    )


def _fold_metric_row(
    y: np.ndarray,
    prediction: np.ndarray,
    validation: np.ndarray,
    *,
    fold: int,
    model_name: str,
    protocol: str,
    shots: int,
    shot_seed: int,
    ridge_alpha: float,
    early_lambda: float,
    transition_lambda: float,
    design_feature_width: int,
) -> dict[str, object]:
    return {
        "fold": int(fold),
        "model_name": model_name,
        "protocol": protocol,
        "shots": int(shots),
        "shot_seed": int(shot_seed),
        "validation_rows": int(validation.sum()),
        "ridge_alpha": float(ridge_alpha),
        "early_lambda": float(early_lambda),
        "transition_lambda": float(transition_lambda),
        "design_feature_width": int(design_feature_width),
        **_metric_payload(y, prediction, validation),
    }


def _evaluate_model(
    matrix: np.ndarray,
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    validation: np.ndarray,
    model_name: str,
    readout_kind: str,
    protocol: str,
    shots: int,
    shot_seed: int,
    config: LadderReadoutUpgradeConfig,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    diagnostics, candidates, correction, design_width = select_inner_configuration(
        matrix,
        y=y,
        har=har,
        residuals=residuals,
        residual_train_mask=residual_train_mask,
        origin_date=frame["origin_date"].astype(str).to_numpy(),
        readout_kind=readout_kind,
        calibration_kind="segmented",
        config=config,
    )
    prediction = apply_calibration(
        har,
        correction,
        early_lambda=float(diagnostics["early_lambda"]),
        transition_lambda=float(diagnostics["transition_lambda"]),
        split_horizon=config.split_horizon,
    )
    metric_row = _fold_metric_row(
        y,
        prediction,
        validation,
        fold=int(frame["fold"].iloc[0]),
        model_name=model_name,
        protocol=protocol,
        shots=shots,
        shot_seed=shot_seed,
        ridge_alpha=float(diagnostics["ridge_alpha"]),
        early_lambda=float(diagnostics["early_lambda"]),
        transition_lambda=float(diagnostics["transition_lambda"]),
        design_feature_width=design_width,
    )
    candidates = candidates.copy()
    candidates.insert(0, "fold", int(frame["fold"].iloc[0]))
    candidates.insert(1, "model_name", model_name)
    candidates.insert(2, "protocol", protocol)
    candidates.insert(3, "shots", int(shots))
    candidates.insert(4, "shot_seed", int(shot_seed))
    predictions = _prediction_frame(
        frame,
        y,
        prediction,
        har,
        validation,
        model_name=model_name,
        protocol=protocol,
        shots=shots,
        shot_seed=shot_seed,
        ridge_alpha=float(diagnostics["ridge_alpha"]),
        early_lambda=float(diagnostics["early_lambda"]),
        transition_lambda=float(diagnostics["transition_lambda"]),
    )
    return metric_row, candidates, predictions


def _pooled_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["model_name", "protocol", "shots", "shot_seed"]
    for values, group in predictions.groupby(keys, sort=True, dropna=False):
        observed = group["y_true"].to_numpy(dtype=float)[:, None]
        forecast = group["y_pred"].to_numpy(dtype=float)[:, None]
        mask = np.ones(len(group), dtype=bool)
        qlike, rmse = metrics(observed, forecast, mask)
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "prediction_rows": int(len(group)),
                "qlike": float(qlike),
                "rmse": float(rmse),
                "mean_error": float(np.mean(forecast - observed)),
                "correlation": _pearson_correlation(observed, forecast),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result

    exact = predictions.loc[
        predictions["protocol"].eq("exact")
        & ~predictions["model_name"].eq("har")
    ][["model_name", "fold", "sample_id", "horizon", "y_pred"]].rename(
        columns={"y_pred": "exact_y_pred"}
    )
    correlations: list[dict[str, object]] = []
    shot_predictions = predictions.loc[
        ~predictions["protocol"].eq("exact")
    ]
    for values, group in shot_predictions.groupby(keys, sort=True):
        merged = group.merge(
            exact,
            on=["model_name", "fold", "sample_id", "horizon"],
            how="inner",
            validate="one_to_one",
        )
        correlations.append(
            {
                **dict(zip(keys, values, strict=True)),
                "exact_prediction_rows": int(len(merged)),
                "prediction_pearson_exact": _pearson_correlation(
                    merged["y_pred"], merged["exact_y_pred"]
                ),
                "prediction_spearman_exact": _rank_correlation(
                    merged["y_pred"], merged["exact_y_pred"]
                ),
            }
        )
    if correlations:
        result = result.merge(
            pd.DataFrame(correlations),
            on=keys,
            how="left",
            validate="one_to_one",
        )
    return result


def _group_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "model_name",
        "protocol",
        "shots",
        "shot_seed",
        "lead",
        "label",
    ]
    for values, group in predictions.groupby(keys, sort=True, dropna=False):
        observed = group["y_true"].to_numpy(dtype=float)[:, None]
        forecast = group["y_pred"].to_numpy(dtype=float)[:, None]
        qlike, rmse = metrics(
            observed,
            forecast,
            np.ones(len(group), dtype=bool),
        )
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "prediction_rows": int(len(group)),
                "qlike": float(qlike),
                "rmse": float(rmse),
                "mean_error": float(np.mean(forecast - observed)),
            }
        )
    return pd.DataFrame(rows)


def summarize_shot_metrics(pooled_metrics: pd.DataFrame) -> pd.DataFrame:
    shot = pooled_metrics.loc[
        ~pooled_metrics["protocol"].eq("exact")
    ].copy()
    if shot.empty:
        return pd.DataFrame()
    keys = ["model_name", "protocol", "shots"]
    rows: list[dict[str, object]] = []
    for values, group in shot.groupby(keys, sort=True):
        row: dict[str, object] = dict(zip(keys, values, strict=True))
        row["seeds"] = int(group["shot_seed"].nunique())
        for metric in (
            "qlike",
            "rmse",
            "prediction_pearson_exact",
            "prediction_spearman_exact",
        ):
            if metric not in group:
                continue
            series = group[metric].dropna().astype(float)
            row[f"{metric}_median"] = (
                float(series.median()) if not series.empty else float("nan")
            )
            row[f"{metric}_minimum"] = (
                float(series.min()) if not series.empty else float("nan")
            )
            row[f"{metric}_maximum"] = (
                float(series.max()) if not series.empty else float("nan")
            )
            row[f"{metric}_std"] = (
                float(series.std(ddof=1)) if len(series) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_shot_degradation(
    summary: pd.DataFrame,
    exact_reference: pd.DataFrame,
    output_path: Path,
) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for (model_name, protocol), group in summary.groupby(
        ["model_name", "protocol"], sort=True
    ):
        ordered = group.sort_values("shots")
        ax.plot(
            ordered["shots"],
            ordered["qlike_median"],
            marker="o",
            label=f"{model_name} | {protocol}",
        )
        exact = exact_reference.loc[
            exact_reference["model_name"].eq(model_name)
        ]
        if not exact.empty:
            ax.axhline(
                float(exact.iloc[0]["qlike"]),
                linestyle="--",
                linewidth=1.0,
            )
    har = exact_reference.loc[exact_reference["model_name"].eq("har")]
    if not har.empty:
        ax.axhline(
            float(har.iloc[0]["qlike"]),
            linestyle=":",
            linewidth=1.5,
            label="HAR exact",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Shots per sample and probe")
    ax.set_ylabel("Pooled validation QLIKE")
    ax.set_title("Finite-shot degradation of frozen segmented ladder readouts")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_ladder_finite_shot_study(
    *,
    fold_dir: Path,
    results_root: Path,
    cache_root: Path,
    config: LadderFiniteShotStudyConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    run_id: str | None = None,
    force_cache: bool = False,
) -> Path:
    """Run exact-reference and finite-shot validation on both segmented heads."""

    config.validate()
    candidate_features.validate()
    reservoir.validate()
    ladder_geometry.validate()
    readout_config = config.readout_config()
    readout_config.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("finite-shot ladder study requires six atoms")
    if reservoir.shots is not None:
        raise ValueError("set reservoir.shots=None; the study samples cached probabilities")
    if tuple(config.folds) != tuple(readout_config.folds):
        raise RuntimeError("study and readout fold contracts differ")

    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "cache_root": str(cache_root),
            "study": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "ladder_geometry": ladder_geometry.to_dict(),
            "models": [name for name, _ in MODEL_SPECS],
            "test_rows_allowed": False,
            "control_manifest_status": "current_pre_redesign_manifest",
            "authoritative_status": "provisional_until_controls_and_geometry_frozen",
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(fold_dir)
    available_folds = set(dataset.manifest["fold"].astype(int).unique())
    missing = set(config.folds).difference(available_folds)
    if missing:
        raise ValueError(
            f"requested folds are absent: {sorted(missing)}; "
            f"available={sorted(available_folds)}"
        )
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )

    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.leads,
            max_per_class=config.max_per_class,
            seed=config.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("finite-shot study must not receive test rows")

        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        source = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=config.sequence_length,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(source).all(axis=1)
        frame = frame.loc[usable].reset_index(drop=True)
        source = source[usable]
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no usable rows remain")

        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        if not train.any() or not validation.any():
            raise RuntimeError(f"fold {fold}: empty train or validation split")

        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train)
        residuals, residual_train = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=config.prequential_blocks,
        )
        raw_sequence = build_candidate_sequences(
            source,
            "level_instability",
            candidate_features,
        )
        scaler = fit_channel_scaler(raw_sequence, train)
        encoded = transform_candidate_sequences(raw_sequence, scaler)
        probe_steps = _resolve_probe_steps(
            config.sequence_length,
            reservoir.probe_fractions,
        )
        identity = cache_identity(
            fold=int(fold),
            sample_ids=frame["sample_id"].astype(str).to_numpy(),
            encoded_windows=encoded,
            reservoir=reservoir,
            geometry=ladder_geometry,
            interaction_scale=config.ladder_interaction_scale,
            probe_steps=probe_steps,
            selection_parameters={
                "fold": int(fold),
                "folds": list(config.folds),
                "leads": list(config.leads),
                "max_per_class": int(config.max_per_class),
                "sequence_length": int(config.sequence_length),
                "selection_seed": int(config.seed),
                "candidate_features": candidate_features.to_dict(),
                "channel_scaler": scaler.to_dict(),
            },
        )
        identity_hash = payload_sha256(identity)
        cache_path = (
            Path(cache_root)
            / f"fold_{int(fold)}"
            / f"{identity_hash[:20]}.npz"
        )
        if force_cache and cache_path.exists():
            cache_path.unlink()
        if cache_path.exists():
            cache = load_probability_cache(
                cache_path,
                expected_identity=identity,
            )
            cache_status = "reused"
        else:
            probabilities, probability_metadata = (
                evolve_ladder_probe_probabilities(
                    encoded,
                    reservoir,
                    ladder_geometry,
                    interaction_scale=config.ladder_interaction_scale,
                    condition="ordered",
                )
            )
            emitted_probe_steps = tuple(
                int(value) for value in probability_metadata["probe_steps"]
            )
            if emitted_probe_steps != tuple(probe_steps):
                raise RuntimeError("probability extractor changed the probe contract")
            exact_modes = probabilities_to_symmetric_modes(probabilities)
            write_probability_cache(
                cache_path,
                probabilities=probabilities,
                exact_modes=exact_modes,
                sample_ids=frame["sample_id"].astype(str).to_numpy(),
                fold_splits=frame["fold_split"].astype(str).to_numpy(),
                leads=frame["lead"].to_numpy(dtype=int),
                labels=frame["label"].to_numpy(dtype=int),
                episode_ids=frame["episode_id"].astype(str).to_numpy(),
                origin_dates=frame["origin_date"].astype(str).to_numpy(),
                identity=identity,
            )
            cache = load_probability_cache(
                cache_path,
                expected_identity=identity,
            )
            cache_status = "created"

        if not np.array_equal(
            cache["sample_id"],
            frame["sample_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError("probability cache sample order differs from selected frame")
        probabilities = np.asarray(cache["probabilities"], dtype=float)
        exact_modes = np.asarray(cache["exact_modes"], dtype=float)
        cache_rows.append(
            {
                "fold": int(fold),
                "cache_status": cache_status,
                "cache_path": str(cache_path),
                "cache_identity_sha256": str(cache["identity_sha256"]),
                "rows": int(len(frame)),
                "probes": int(probabilities.shape[1]),
                "states": int(probabilities.shape[2]),
                "exact_mode_width": int(exact_modes.shape[1]),
            }
        )

        har_row = _fold_metric_row(
            y,
            har,
            validation,
            fold=int(fold),
            model_name="har",
            protocol="exact",
            shots=0,
            shot_seed=-1,
            ridge_alpha=0.0,
            early_lambda=0.0,
            transition_lambda=0.0,
            design_feature_width=0,
        )
        fold_metric_rows.append(har_row)
        prediction_frames.append(
            _prediction_frame(
                frame,
                y,
                har,
                har,
                validation,
                model_name="har",
                protocol="exact",
                shots=0,
                shot_seed=-1,
                ridge_alpha=0.0,
                early_lambda=0.0,
                transition_lambda=0.0,
            )
        )

        for model_name, readout_kind in MODEL_SPECS:
            metric_row, candidates, predictions = _evaluate_model(
                exact_modes,
                frame=frame,
                y=y,
                har=har,
                residuals=residuals,
                residual_train_mask=residual_train,
                validation=validation,
                model_name=model_name,
                readout_kind=readout_kind,
                protocol="exact",
                shots=0,
                shot_seed=-1,
                config=readout_config,
            )
            fold_metric_rows.append(metric_row)
            candidate_frames.append(candidates)
            prediction_frames.append(predictions)

        for shots in config.shot_counts:
            for shot_seed in config.shot_seeds:
                sampled_modes = shot_modes_from_probabilities(
                    probabilities,
                    shots=int(shots),
                    base_seed=int(shot_seed),
                    fold=int(fold),
                    sample_ids=frame["sample_id"].astype(str).to_numpy(),
                    probe_steps=probe_steps,
                )
                feature_rows.append(
                    {
                        "fold": int(fold),
                        "shots": int(shots),
                        "shot_seed": int(shot_seed),
                        "rows": int(len(sampled_modes)),
                        "feature_width": int(sampled_modes.shape[1]),
                        "mode_rmse_all": float(
                            np.sqrt(np.mean((sampled_modes - exact_modes) ** 2))
                        ),
                        "mode_pearson_all": _pearson_correlation(
                            sampled_modes, exact_modes
                        ),
                        "mode_spearman_all": _rank_correlation(
                            sampled_modes, exact_modes
                        ),
                        "mode_rmse_validation": float(
                            np.sqrt(
                                np.mean(
                                    (
                                        sampled_modes[validation]
                                        - exact_modes[validation]
                                    )
                                    ** 2
                                )
                            )
                        ),
                    }
                )
                for protocol in config.protocols:
                    matrix = protocol_feature_matrix(
                        exact_modes=exact_modes,
                        sampled_modes=sampled_modes,
                        train_mask=train,
                        protocol=protocol,
                    )
                    for model_name, readout_kind in MODEL_SPECS:
                        metric_row, candidates, predictions = _evaluate_model(
                            matrix,
                            frame=frame,
                            y=y,
                            har=har,
                            residuals=residuals,
                            residual_train_mask=residual_train,
                            validation=validation,
                            model_name=model_name,
                            readout_kind=readout_kind,
                            protocol=protocol,
                            shots=int(shots),
                            shot_seed=int(shot_seed),
                            config=readout_config,
                        )
                        fold_metric_rows.append(metric_row)
                        candidate_frames.append(candidates)
                        prediction_frames.append(predictions)

    fold_metrics = pd.DataFrame(fold_metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    inner_candidates = pd.concat(candidate_frames, ignore_index=True)
    feature_stability = pd.DataFrame(feature_rows)
    cache_manifest = pd.DataFrame(cache_rows)
    pooled_metrics = _pooled_metric_table(predictions)
    exact_reference = pooled_metrics.loc[
        pooled_metrics["protocol"].eq("exact")
    ].reset_index(drop=True)
    shot_metrics = pooled_metrics.loc[
        ~pooled_metrics["protocol"].eq("exact")
    ].reset_index(drop=True)
    shot_summary = summarize_shot_metrics(pooled_metrics)
    group_metrics = _group_metric_table(predictions)

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    exact_reference.to_csv(run_dir / "exact_reference_metrics.csv", index=False)
    shot_metrics.to_csv(run_dir / "shot_metrics_by_seed.csv", index=False)
    shot_summary.to_csv(run_dir / "shot_metrics_summary.csv", index=False)
    group_metrics.to_csv(run_dir / "group_metrics_by_seed.csv", index=False)
    feature_stability.to_csv(run_dir / "feature_stability.csv", index=False)
    cache_manifest.to_csv(run_dir / "cache_manifest.csv", index=False)
    inner_candidates.to_csv(
        run_dir / "inner_candidates.csv.gz",
        index=False,
        compression="gzip",
    )
    predictions.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    calibration_columns = [
        "fold",
        "model_name",
        "protocol",
        "shots",
        "shot_seed",
        "ridge_alpha",
        "early_lambda",
        "transition_lambda",
    ]
    fold_metrics[calibration_columns].to_csv(
        run_dir / "calibration_stability.csv",
        index=False,
    )
    _plot_shot_degradation(
        shot_summary,
        exact_reference,
        run_dir / "shot_degradation_curve.png",
    )

    summary = {
        "schema_version": 1,
        "status": "finite_shot_study_complete",
        "evaluation_split": "validation",
        "folds": list(config.folds),
        "shot_counts": list(config.shot_counts),
        "shot_seeds": list(config.shot_seeds),
        "protocols": list(config.protocols),
        "models": [name for name, _ in MODEL_SPECS],
        "test_rows_used": 0,
        "probability_cache": {
            "schema_version": 1,
            "rows": int(len(cache_manifest)),
            "created": int(cache_manifest["cache_status"].eq("created").sum()),
            "reused": int(cache_manifest["cache_status"].eq("reused").sum()),
        },
        "measurement_definition": (
            "one independent multinomial ensemble per sample and probe, with "
            "stable hash-derived seeds invariant to row ordering"
        ),
        "primary_protocol": "shot_consistent",
        "diagnostic_protocol": "exact_train_shot_val",
        "control_manifest_status": "current_pre_redesign_manifest",
        "authoritative_status": (
            "provisional until deterministic controls and final geometry are frozen"
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir
