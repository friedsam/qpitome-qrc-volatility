from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.qrc.frozen_ladder_confirmation_tools import (
    symmetric_ladder_mode_matrix,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    build_candidate_sequences,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import (
    feature_names_from_metadata,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    build_temporal_rydberg_ladder_features,
)


_TICKER_RE = re.compile(r"(?:^|_)(\^?[A-Za-z0-9.]+)_data_L(?:1|5|10)(?:_|$)")
_REPRESENTATIONS = ("level_instability", "level_downside_return")
_CHANNEL_NAMES = ("level", "second")


@dataclass(frozen=True)
class InputSensitivityRankConfig:
    """Mechanistic assay for the controllable rank of the frozen ladder map."""

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    lead: int = 5
    max_per_class: int = 4
    sequence_length: int = 40
    perturb_lags: tuple[int, ...] = (0, 1, 2, 4, 9, 19, 39)
    perturbation_epsilon: float = 0.05
    reconstruction_alpha: float = 1.0
    channel_subspace_energy: float = 0.95
    seed: int = 20260724
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds:
            raise ValueError("folds cannot be empty")
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be unique")
        if self.lead < 1:
            raise ValueError("lead must be positive")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if not self.perturb_lags:
            raise ValueError("perturb_lags cannot be empty")
        if len(set(self.perturb_lags)) != len(self.perturb_lags):
            raise ValueError("perturb_lags must be unique")
        if any(lag < 0 or lag >= self.sequence_length for lag in self.perturb_lags):
            raise ValueError("perturb_lags must lie in [0, sequence_length)")
        if not 0.0 < self.perturbation_epsilon < 1.0:
            raise ValueError("perturbation_epsilon must lie in (0, 1)")
        if self.reconstruction_alpha <= 0.0:
            raise ValueError("reconstruction_alpha must be positive")
        if not 0.0 < self.channel_subspace_energy <= 1.0:
            raise ValueError("channel_subspace_energy must lie in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationBlock:
    channel: int
    lag: int
    plus: slice
    minus: slice
    denominator: np.ndarray


def parse_ticker(sample_id: str) -> str:
    match = _TICKER_RE.search(str(sample_id))
    if match is None:
        raise ValueError(f"cannot parse ticker from sample_id={sample_id!r}")
    return match.group(1)


def load_close_panel(path: Path) -> dict[str, pd.Series]:
    frame = pd.read_csv(path, usecols=["date", "close", "ticker"])
    frame["date"] = pd.to_datetime(
        frame["date"], format="mixed", utc=True, errors="raise"
    ).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(
        ["ticker", "date"], keep="last"
    )
    output: dict[str, pd.Series] = {}
    for ticker, group in frame.groupby("ticker", sort=True):
        series = group.set_index("date")["close"].sort_index()
        output[str(ticker)] = series
    return output


def downside_return_windows(
    frame: pd.DataFrame,
    panel: dict[str, pd.Series],
    *,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.full((len(frame), sequence_length), np.nan, dtype=float)
    valid = np.zeros(len(frame), dtype=bool)
    origin_dates = pd.to_datetime(
        frame["origin_date"], format="mixed", utc=True, errors="raise"
    ).dt.normalize()
    for row_index, row in frame.iterrows():
        ticker = parse_ticker(str(row["sample_id"]))
        if ticker not in panel:
            continue
        history = (
            panel[ticker]
            .loc[: origin_dates.iloc[row_index]]
            .dropna()
            .tail(sequence_length + 1)
        )
        values = history.to_numpy(dtype=float)
        if (
            len(values) != sequence_length + 1
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
        ):
            continue
        returns = np.diff(np.log(values))
        windows[row_index] = np.minimum(returns, 0.0)
        valid[row_index] = True
    return windows, valid


def build_perturbation_batch(
    encoded_sequences: np.ndarray,
    *,
    lags: tuple[int, ...],
    epsilon: float,
) -> tuple[np.ndarray, tuple[PerturbationBlock, ...]]:
    sequences = np.asarray(encoded_sequences, dtype=float)
    if (
        sequences.ndim != 3
        or sequences.shape[2] != 2
        or not np.isfinite(sequences).all()
    ):
        raise ValueError("encoded_sequences must be finite with shape (samples, time, 2)")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    samples, steps, _ = sequences.shape
    if any(lag < 0 or lag >= steps for lag in lags):
        raise ValueError("lags must lie inside the encoded sequence")

    batches = [sequences]
    records: list[PerturbationBlock] = []
    cursor = samples
    for channel in range(2):
        for lag in lags:
            step = steps - 1 - int(lag)
            plus = sequences.copy()
            minus = sequences.copy()
            plus[:, step, channel] = np.clip(
                plus[:, step, channel] + float(epsilon), -1.0, 1.0
            )
            minus[:, step, channel] = np.clip(
                minus[:, step, channel] - float(epsilon), -1.0, 1.0
            )
            denominator = plus[:, step, channel] - minus[:, step, channel]
            if np.any(denominator <= 0.0):
                raise RuntimeError("finite-difference perturbation has zero denominator")
            plus_slice = slice(cursor, cursor + samples)
            cursor += samples
            minus_slice = slice(cursor, cursor + samples)
            cursor += samples
            batches.extend([plus, minus])
            records.append(
                PerturbationBlock(
                    channel=int(channel),
                    lag=int(lag),
                    plus=plus_slice,
                    minus=minus_slice,
                    denominator=denominator,
                )
            )
    return np.concatenate(batches, axis=0), tuple(records)


def finite_difference_sensitivity(
    feature_matrix: np.ndarray,
    *,
    samples: int,
    lags: tuple[int, ...],
    blocks: tuple[PerturbationBlock, ...],
) -> np.ndarray:
    features = np.asarray(feature_matrix, dtype=float)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("feature_matrix must be finite and two-dimensional")
    output = np.empty((samples, 2, len(lags), features.shape[1]), dtype=float)
    lag_index = {int(lag): index for index, lag in enumerate(lags)}
    for block in blocks:
        numerator = features[block.plus] - features[block.minus]
        output[:, block.channel, lag_index[block.lag], :] = (
            numerator / block.denominator[:, None]
        )
    return output


def _rank_payload(matrix: np.ndarray, *, standardize: bool) -> dict[str, object]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        return {
            "rows": int(len(values)) if values.ndim >= 1 else 0,
            "columns": int(values.shape[1]) if values.ndim == 2 else 0,
            "active_columns": 0,
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "pc1_power_share": np.nan,
            "pc2_cumulative_power_share": np.nan,
            "pc3_cumulative_power_share": np.nan,
            "singular_values": "[]",
        }
    centered = values - values.mean(axis=0, keepdims=True)
    column_std = centered.std(axis=0)
    active = column_std > 1e-12
    centered = centered[:, active]
    if centered.shape[1] == 0:
        return {
            "rows": int(len(values)),
            "columns": int(values.shape[1]),
            "active_columns": 0,
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "pc1_power_share": np.nan,
            "pc2_cumulative_power_share": np.nan,
            "pc3_cumulative_power_share": np.nan,
            "singular_values": "[]",
        }
    if standardize:
        centered = centered / column_std[active][None, :]
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular**2
    total = float(power.sum())
    if total <= 0.0:
        probabilities = np.zeros_like(power)
        effective = 0.0
    else:
        probabilities = power / total
        positive = probabilities > 0.0
        effective = float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
    cumulative = np.cumsum(probabilities)
    return {
        "rows": int(len(values)),
        "columns": int(values.shape[1]),
        "active_columns": int(active.sum()),
        "numerical_rank": int(np.linalg.matrix_rank(centered)),
        "effective_rank": effective,
        "pc1_power_share": float(probabilities[0]) if len(probabilities) else np.nan,
        "pc2_cumulative_power_share": (
            float(cumulative[min(1, len(cumulative) - 1)]) if len(cumulative) else np.nan
        ),
        "pc3_cumulative_power_share": (
            float(cumulative[min(2, len(cumulative) - 1)]) if len(cumulative) else np.nan
        ),
        "singular_values": json.dumps([float(value) for value in singular]),
    }


def channel_novelty_fraction(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    energy: float = 0.95,
) -> tuple[float, int]:
    """Fraction of candidate response outside the dominant reference subspace."""

    left = np.asarray(reference, dtype=float)
    right = np.asarray(candidate, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("reference and candidate must be aligned matrices")
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-15:
        return np.nan, 0
    left_norm = float(np.linalg.norm(left))
    if left_norm <= 1e-15:
        return 1.0, 0
    _, singular, right_vectors = np.linalg.svd(left, full_matrices=False)
    power = singular**2
    cumulative = np.cumsum(power) / float(power.sum())
    rank = int(np.searchsorted(cumulative, float(energy), side="left") + 1)
    basis = right_vectors[:rank]
    projection = (right @ basis.T) @ basis
    residual = right - projection
    return float(np.linalg.norm(residual) / right_norm), rank


def matched_channel_cosines(sensitivity: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(sensitivity, dtype=float)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("sensitivity must have shape (samples, 2, lags, features)")
    left = values[:, 0].reshape(-1, values.shape[-1])
    right = values[:, 1].reshape(-1, values.shape[-1])
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 1e-15
    if not valid.any():
        return np.nan, np.nan, 0
    cosine = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    absolute = np.abs(cosine)
    return float(absolute.mean()), float(np.median(absolute)), int(valid.sum())


def _feature_views(
    full_matrix: np.ndarray,
    feature_names: tuple[str, ...],
    probe_steps: tuple[int, ...],
) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    values = np.asarray(full_matrix, dtype=float)

    def selected(name: str, predicate: object) -> tuple[np.ndarray, tuple[str, ...]]:
        indices = [
            index
            for index, feature_name in enumerate(feature_names)
            if predicate(feature_name)
        ]
        if not indices:
            raise ValueError(f"feature view {name!r} selected no columns")
        return values[:, indices], tuple(feature_names[index] for index in indices)

    symmetric, symmetric_names = symmetric_ladder_mode_matrix(
        values, feature_names, probe_steps
    )
    return {
        "full_observables": (values, feature_names),
        "occupations": selected(
            "occupations", lambda value: "_occupation_site_" in value
        ),
        "pair_observables": selected(
            "pair_observables",
            lambda value: (
                "_nearest_pair_" in value
                or "_nearest_connected_" in value
                or "_long_pair_" in value
                or "_long_connected_" in value
            ),
        ),
        "connected_correlations": selected(
            "connected_correlations",
            lambda value: (
                "_nearest_connected_" in value or "_long_connected_" in value
            ),
        ),
        "symmetric_modes": (symmetric, symmetric_names),
    }


def _probe_steps_from_names(names: tuple[str, ...]) -> tuple[int, ...]:
    observed: set[int] = set()
    for name in names:
        match = re.match(r"probe_(\d+)_", name)
        if match:
            observed.add(int(match.group(1)))
    return tuple(sorted(observed))


def _lag_sensitivity_rows(
    sensitivity: np.ndarray,
    *,
    representation: str,
    view: str,
    feature_names: tuple[str, ...],
    lags: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    probes = _probe_steps_from_names(feature_names)
    masks: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(feature_names), dtype=bool))
    ]
    for probe in probes:
        masks.append(
            (
                str(probe),
                np.asarray(
                    [name.startswith(f"probe_{probe}_") for name in feature_names],
                    dtype=bool,
                ),
            )
        )
    for channel in range(2):
        for lag_index, lag in enumerate(lags):
            current = sensitivity[:, channel, lag_index, :]
            for probe, mask in masks:
                local = current[:, mask]
                norms = np.linalg.norm(local, axis=1)
                rows.append(
                    {
                        "representation": representation,
                        "view": view,
                        "channel": _CHANNEL_NAMES[channel],
                        "channel_index": int(channel),
                        "lag": int(lag),
                        "probe": probe,
                        "features": int(mask.sum()),
                        "mean_l2_sensitivity": float(norms.mean()),
                        "median_l2_sensitivity": float(np.median(norms)),
                        "maximum_l2_sensitivity": float(norms.max(initial=0.0)),
                        "near_zero_fraction": float((norms < 1e-10).mean()),
                    }
                )
    return rows


def _reconstruction_rows(
    features: np.ndarray,
    encoded_sequences: np.ndarray,
    frame: pd.DataFrame,
    *,
    representation: str,
    view: str,
    lags: tuple[int, ...],
    alpha: float,
) -> list[dict[str, object]]:
    matrix = np.asarray(features, dtype=float)
    encoded = np.asarray(encoded_sequences, dtype=float)
    rows: list[dict[str, object]] = []
    for fold in sorted(frame["fold"].astype(int).unique()):
        train = (
            frame["fold"].eq(fold).to_numpy()
            & frame["fold_split"].eq("train").to_numpy()
        )
        validation = (
            frame["fold"].eq(fold).to_numpy()
            & frame["fold_split"].eq("val").to_numpy()
        )
        if train.sum() < 4 or validation.sum() < 2:
            continue
        scaler = StandardScaler().fit(matrix[train])
        train_design = scaler.transform(matrix[train])
        validation_design = scaler.transform(matrix[validation])
        for channel in range(2):
            for lag in lags:
                step = encoded.shape[1] - 1 - int(lag)
                target = encoded[:, step, channel]
                model = Ridge(alpha=float(alpha)).fit(train_design, target[train])
                prediction = model.predict(validation_design)
                observed = target[validation]
                denominator = float(np.sum((observed - observed.mean()) ** 2))
                r2 = (
                    float(1.0 - np.sum((observed - prediction) ** 2) / denominator)
                    if denominator > 1e-15
                    else np.nan
                )
                correlation = (
                    float(np.corrcoef(observed, prediction)[0, 1])
                    if np.std(observed) > 1e-15 and np.std(prediction) > 1e-15
                    else np.nan
                )
                rows.append(
                    {
                        "representation": representation,
                        "view": view,
                        "fold": int(fold),
                        "channel": _CHANNEL_NAMES[channel],
                        "channel_index": int(channel),
                        "lag": int(lag),
                        "train_rows": int(train.sum()),
                        "validation_rows": int(validation.sum()),
                        "ridge_alpha": float(alpha),
                        "r2": r2,
                        "correlation": correlation,
                        "rmse": float(np.sqrt(np.mean((observed - prediction) ** 2))),
                    }
                )
    return rows


def _prepare_encoded_sequences(
    *,
    fold_dir: Path,
    close_panel_path: Path,
    config: InputSensitivityRankConfig,
    candidate_features: CandidateFeatureConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, object]]]:
    dataset = load_rolling_fold_dataset(fold_dir)
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )
    panel = load_close_panel(close_panel_path)
    frame_blocks: list[pd.DataFrame] = []
    encoded_blocks: dict[str, list[np.ndarray]] = {
        representation: [] for representation in _REPRESENTATIONS
    }
    scaler_rows: list[dict[str, object]] = []

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=(int(config.lead),),
            max_per_class=int(config.max_per_class),
            seed=int(config.seed),
        )
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        level = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=config.sequence_length,
            level_channel=level_channel,
        )
        downside, downside_valid = downside_return_windows(
            frame,
            panel,
            sequence_length=config.sequence_length,
        )
        usable = (
            dataset.valid[tensor_rows]
            & np.isfinite(level).all(axis=1)
            & downside_valid
        )
        frame = frame.loc[usable].reset_index(drop=True)
        level = level[usable]
        downside = downside[usable]
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no rows have aligned level and downside histories")
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        if not train.any() or not validation.any():
            raise RuntimeError(f"fold {fold}: empty train or validation split after alignment")

        raw_representations = {
            "level_instability": build_candidate_sequences(
                level, "level_instability", candidate_features
            ),
            "level_downside_return": np.stack([level, downside], axis=-1),
        }
        for representation, raw in raw_representations.items():
            scaler = fit_channel_scaler(raw, train)
            encoded = transform_candidate_sequences(raw, scaler)
            encoded_blocks[representation].append(encoded)
            scaler_rows.append(
                {
                    "fold": int(fold),
                    "representation": representation,
                    "train_rows": int(train.sum()),
                    "validation_rows": int(validation.sum()),
                    "channel_0_median": float(scaler.medians[0]),
                    "channel_1_median": float(scaler.medians[1]),
                    "channel_0_half_range": float(scaler.half_ranges[0]),
                    "channel_1_half_range": float(scaler.half_ranges[1]),
                }
            )
        frame_blocks.append(
            frame[
                [
                    "fold",
                    "fold_split",
                    "sample_id",
                    "lead",
                    "label",
                    "episode_id",
                    "origin_date",
                ]
            ].copy()
        )

    combined_frame = pd.concat(frame_blocks, ignore_index=True)
    if combined_frame["fold_split"].eq("test").any():
        raise RuntimeError("sensitivity assay must not receive test rows")
    combined_encoded = {
        representation: np.concatenate(blocks, axis=0)
        for representation, blocks in encoded_blocks.items()
    }
    expected = len(combined_frame)
    if any(len(values) != expected for values in combined_encoded.values()):
        raise RuntimeError("encoded representations are not aligned to the sample frame")
    return combined_frame, combined_encoded, scaler_rows


def run_input_sensitivity_rank_assay(
    *,
    fold_dir: Path,
    close_panel_path: Path,
    results_root: Path,
    config: InputSensitivityRankConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
    run_id: str,
) -> Path:
    """Measure channel/lag sensitivity, output rank, and input reconstructability."""

    config.validate()
    candidate_features.validate()
    reservoir.validate()
    ladder_geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("the staggered ladder sensitivity assay requires six atoms")
    if reservoir.shots is not None:
        raise ValueError("the mechanistic sensitivity assay requires exact-state features")
    if interaction_scale <= 0.0:
        raise ValueError("interaction_scale must be positive")

    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "close_panel_path": str(close_panel_path),
            "config": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "ladder_geometry": ladder_geometry.to_dict(),
            "interaction_scale": float(interaction_scale),
            "scientific_question": (
                "Does replacing volatility-derived instability with downside return create "
                "a more independent, memory-bearing, measurable second control direction "
                "in the frozen six-atom ladder?"
            ),
            "forecast_selection_performed": False,
            "readout_optimization_performed": False,
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    frame, encoded_by_representation, scaler_rows = _prepare_encoded_sequences(
        fold_dir=fold_dir,
        close_panel_path=close_panel_path,
        config=config,
        candidate_features=candidate_features,
    )
    frame.to_csv(run_dir / "selected_samples.csv", index=False)
    pd.DataFrame(scaler_rows).to_csv(run_dir / "channel_scalers.csv", index=False)

    feature_rank_rows: list[dict[str, object]] = []
    sensitivity_rank_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    lag_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    archive_files: dict[str, str] = {}

    for representation in _REPRESENTATIONS:
        encoded = encoded_by_representation[representation]
        perturbation_batch, perturbation_blocks = build_perturbation_batch(
            encoded,
            lags=config.perturb_lags,
            epsilon=config.perturbation_epsilon,
        )
        full_features, metadata = build_temporal_rydberg_ladder_features(
            perturbation_batch,
            reservoir,
            ladder_geometry,
            interaction_scale=float(interaction_scale),
            condition="ordered",
        )
        feature_names = feature_names_from_metadata(metadata)
        probe_steps = tuple(int(value) for value in metadata["probe_steps"])
        views = _feature_views(full_features, feature_names, probe_steps)
        archive_payload: dict[str, np.ndarray] = {
            "encoded_sequences": encoded,
            "perturb_lags": np.asarray(config.perturb_lags, dtype=int),
            "fold": frame["fold"].to_numpy(dtype=int),
            "fold_split": frame["fold_split"].astype(str).to_numpy(),
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "lead": frame["lead"].to_numpy(dtype=int),
            "label": frame["label"].to_numpy(dtype=int),
        }

        for view_name, (view_features, view_names) in views.items():
            base = view_features[: len(frame)]
            sensitivity = finite_difference_sensitivity(
                view_features,
                samples=len(frame),
                lags=config.perturb_lags,
                blocks=perturbation_blocks,
            )
            for standardize in (False, True):
                feature_rank_rows.append(
                    {
                        "representation": representation,
                        "view": view_name,
                        "standardized": bool(standardize),
                        **_rank_payload(base, standardize=standardize),
                    }
                )
            flattened = sensitivity.reshape(-1, sensitivity.shape[-1])
            sensitivity_rank_rows.append(
                {
                    "representation": representation,
                    "view": view_name,
                    **_rank_payload(flattened, standardize=False),
                }
            )
            first = sensitivity[:, 0].reshape(-1, sensitivity.shape[-1])
            second = sensitivity[:, 1].reshape(-1, sensitivity.shape[-1])
            novelty, reference_rank = channel_novelty_fraction(
                first,
                second,
                energy=config.channel_subspace_energy,
            )
            mean_cosine, median_cosine, cosine_rows = matched_channel_cosines(
                sensitivity
            )
            first_norm = float(np.linalg.norm(first))
            second_norm = float(np.linalg.norm(second))
            channel_rows.append(
                {
                    "representation": representation,
                    "view": view_name,
                    "reference_channel": "level",
                    "candidate_channel": "second",
                    "reference_subspace_energy": float(config.channel_subspace_energy),
                    "reference_subspace_rank": int(reference_rank),
                    "candidate_novelty_fraction": novelty,
                    "mean_absolute_matched_cosine": mean_cosine,
                    "median_absolute_matched_cosine": median_cosine,
                    "matched_cosine_rows": int(cosine_rows),
                    "level_sensitivity_frobenius_norm": first_norm,
                    "second_sensitivity_frobenius_norm": second_norm,
                    "second_to_level_norm_ratio": (
                        second_norm / first_norm if first_norm > 1e-15 else np.nan
                    ),
                }
            )
            lag_rows.extend(
                _lag_sensitivity_rows(
                    sensitivity,
                    representation=representation,
                    view=view_name,
                    feature_names=view_names,
                    lags=config.perturb_lags,
                )
            )
            reconstruction_rows.extend(
                _reconstruction_rows(
                    base,
                    encoded,
                    frame,
                    representation=representation,
                    view=view_name,
                    lags=config.perturb_lags,
                    alpha=config.reconstruction_alpha,
                )
            )
            safe_name = view_name.replace("/", "_")
            archive_payload[f"base__{safe_name}"] = base
            archive_payload[f"sensitivity__{safe_name}"] = sensitivity
            archive_payload[f"feature_names__{safe_name}"] = np.asarray(
                view_names, dtype=str
            )

        archive_name = f"{representation}_sensitivity.npz"
        np.savez_compressed(run_dir / archive_name, **archive_payload)
        archive_files[representation] = archive_name

    feature_rank = pd.DataFrame(feature_rank_rows)
    sensitivity_rank = pd.DataFrame(sensitivity_rank_rows)
    channel_independence = pd.DataFrame(channel_rows)
    lag_sensitivity = pd.DataFrame(lag_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)

    feature_rank.to_csv(run_dir / "feature_rank.csv", index=False)
    sensitivity_rank.to_csv(run_dir / "sensitivity_rank.csv", index=False)
    channel_independence.to_csv(run_dir / "channel_independence.csv", index=False)
    lag_sensitivity.to_csv(run_dir / "lag_sensitivity.csv", index=False)
    reconstruction.to_csv(run_dir / "input_reconstruction.csv", index=False)

    comparison = channel_independence.pivot(
        index="view",
        columns="representation",
        values=[
            "candidate_novelty_fraction",
            "mean_absolute_matched_cosine",
            "second_to_level_norm_ratio",
        ],
    )
    comparison.columns = [
        f"{metric}__{representation}" for metric, representation in comparison.columns
    ]
    comparison = comparison.reset_index()
    for metric in (
        "candidate_novelty_fraction",
        "mean_absolute_matched_cosine",
        "second_to_level_norm_ratio",
    ):
        incumbent = f"{metric}__level_instability"
        candidate = f"{metric}__level_downside_return"
        if incumbent in comparison.columns and candidate in comparison.columns:
            comparison[f"delta__{metric}"] = comparison[candidate] - comparison[incumbent]
    comparison.to_csv(run_dir / "representation_comparison.csv", index=False)

    reconstruction_summary = (
        reconstruction.groupby(
            ["representation", "view", "channel", "lag"], as_index=False
        )
        .agg(
            folds=("fold", "nunique"),
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            mean_correlation=("correlation", "mean"),
            mean_rmse=("rmse", "mean"),
        )
        .sort_values(["representation", "view", "channel", "lag"])
    )
    reconstruction_summary.to_csv(
        run_dir / "input_reconstruction_summary.csv", index=False
    )

    summary = {
        "schema_version": 1,
        "status": "mechanistic_input_sensitivity_rank_complete",
        "test_rows_used": 0,
        "forecast_metrics_computed": False,
        "readout_hyperparameters_selected": False,
        "samples": int(len(frame)),
        "folds": [int(value) for value in config.folds],
        "lead": int(config.lead),
        "representations": list(_REPRESENTATIONS),
        "views": sorted(channel_independence["view"].unique().tolist()),
        "archives": archive_files,
        "files": {
            "selected_samples": "selected_samples.csv",
            "channel_scalers": "channel_scalers.csv",
            "feature_rank": "feature_rank.csv",
            "sensitivity_rank": "sensitivity_rank.csv",
            "channel_independence": "channel_independence.csv",
            "lag_sensitivity": "lag_sensitivity.csv",
            "input_reconstruction": "input_reconstruction.csv",
            "input_reconstruction_summary": "input_reconstruction_summary.csv",
            "representation_comparison": "representation_comparison.csv",
        },
        "interpretation": (
            "A useful replacement second channel should create a non-negligible response, "
            "a substantial fraction outside the level-response subspace, and reconstructable "
            "lagged information. Higher output rank without channel novelty or reconstructability "
            "is expansion of irrelevant variation rather than improved reservoir processing."
        ),
        "known_limitations": [
            "This assay measures local finite-difference sensitivity around observed histories; "
            "it does not prove global injectivity of the reservoir map.",
            "Input reconstruction is a fixed-alpha diagnostic, not a selected forecast model.",
            "No volatility target or HAR residual is used, so task relevance must be tested only "
            "after the mechanistic gate is understood.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
