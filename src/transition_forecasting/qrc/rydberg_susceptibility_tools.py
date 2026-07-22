from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from transition_forecasting.qrc.ladder_mode_readout_tools import (
    ladder_mode_weights,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _evolve_step_batch,
    _fresh_states,
    _resolve_probe_steps,
    effective_rank,
    encode_drives,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)


@dataclass(frozen=True)
class SusceptibilityAssayConfig:
    """Development-only L5 transition-susceptibility protocol."""

    folds: tuple[int, ...] = tuple(range(1, 9))
    lead: int = 5
    max_per_class: int = 12
    sequence_length: int = 40
    interaction_scale: float = 1.25
    probe_delta_offset_rad_us: float = 0.8
    response_probe_steps: tuple[int, ...] = (1, 3, 5)
    classifier_c: float = 0.1
    shot_budgets: tuple[int, ...] = (1000, 5000)
    shot_replicates: int = 5
    seed: int = 20260722
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or any(int(fold) < 1 for fold in self.folds):
            raise ValueError("folds must be positive and nonempty")
        if self.lead < 1:
            raise ValueError("lead must be positive")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if self.probe_delta_offset_rad_us <= 0:
            raise ValueError("probe_delta_offset_rad_us must be positive")
        if not self.response_probe_steps:
            raise ValueError("response_probe_steps cannot be empty")
        if tuple(sorted(set(self.response_probe_steps))) != self.response_probe_steps:
            raise ValueError("response_probe_steps must be unique and sorted")
        if self.response_probe_steps[0] < 1:
            raise ValueError("response probe steps must be positive")
        if self.classifier_c <= 0:
            raise ValueError("classifier_c must be positive")
        if any(int(shots) < 1 for shots in self.shot_budgets):
            raise ValueError("shot budgets must be positive")
        if self.shot_replicates < 1:
            raise ValueError("shot_replicates must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedLadder:
    """Prepared history state and non-destructive simulator snapshots."""

    final_states: np.ndarray
    history_states: dict[int, np.ndarray]
    final_delta: np.ndarray
    final_omega: np.ndarray
    precomputed: object
    history_probe_steps: tuple[int, ...]


def reporter_geometry(
    geometry: StaggeredLadderGeometryConfig,
    *,
    displaced: bool,
) -> StaggeredLadderGeometryConfig:
    """Return the selected reporter geometry without changing the ladder family."""

    if displaced:
        return geometry
    return replace(geometry, defect_dx_um=0.0, defect_dy_um=0.0)


def prepare_ladder_history(
    scaled_windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float,
) -> PreparedLadder:
    """Prepare a persistent ladder state and retain requested history snapshots."""

    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("the reporter assay requires six atoms")
    if reservoir.shots is not None:
        raise ValueError("prepare_ladder_history requires exact state evolution")
    windows = np.asarray(scaled_windows, dtype=float)
    if (
        windows.ndim != 3
        or windows.shape[2] != 2
        or not np.isfinite(windows).all()
    ):
        raise ValueError("scaled_windows must be finite with shape (samples, time, 2)")

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    delta, omega = encode_drives(windows, reservoir, condition="ordered")
    samples, steps, _ = windows.shape
    probes = _resolve_probe_steps(steps, reservoir.probe_fractions)
    states = _fresh_states(samples, precomputed.n_atoms)
    snapshots: dict[int, np.ndarray] = {}
    for step in range(steps):
        states = _evolve_step_batch(
            states,
            omega[:, step],
            delta[:, step],
            reservoir,
            precomputed,
            interactions=True,
        )
        if step + 1 in probes:
            snapshots[int(step + 1)] = states.copy()
    return PreparedLadder(
        final_states=states.copy(),
        history_states=snapshots,
        final_delta=delta[:, -1].copy(),
        final_omega=omega[:, -1].copy(),
        precomputed=precomputed,
        history_probe_steps=tuple(int(value) for value in probes),
    )


def probe_prepared_ladder(
    prepared: PreparedLadder,
    reservoir: TemporalRydbergChainConfig,
    *,
    delta_offset_rad_us: float,
    response_probe_steps: tuple[int, ...],
) -> dict[str, dict[int, np.ndarray]]:
    """Fork a prepared state into minus, zero and plus detuning probes."""

    if delta_offset_rad_us <= 0:
        raise ValueError("delta_offset_rad_us must be positive")
    if not response_probe_steps:
        raise ValueError("response_probe_steps cannot be empty")
    maximum = max(int(value) for value in response_probe_steps)
    requested = set(int(value) for value in response_probe_steps)
    branch_offsets = {
        "minus": -float(delta_offset_rad_us),
        "zero": 0.0,
        "plus": float(delta_offset_rad_us),
    }
    output: dict[str, dict[int, np.ndarray]] = {}
    for branch, offset in branch_offsets.items():
        states = prepared.final_states.copy()
        branch_states: dict[int, np.ndarray] = {}
        for step in range(1, maximum + 1):
            states = _evolve_step_batch(
                states,
                prepared.final_omega,
                prepared.final_delta + offset,
                reservoir,
                prepared.precomputed,
                interactions=True,
            )
            if step in requested:
                branch_states[int(step)] = states.copy()
        output[branch] = branch_states
    return output


def _occupation_estimate(
    states: np.ndarray,
    precomputed: object,
    *,
    shots: int | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    probabilities = np.abs(np.asarray(states, dtype=complex)) ** 2
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if shots is not None:
        if shots < 1 or rng is None:
            raise ValueError("finite-shot estimation requires shots and rng")
        sampled = np.empty_like(probabilities)
        for row, probs in enumerate(probabilities):
            sampled[row] = rng.multinomial(int(shots), probs) / float(shots)
        probabilities = sampled
    return probabilities @ precomputed.occupation_bits


def susceptibility_feature_blocks(
    prepared: PreparedLadder,
    branches: dict[str, dict[int, np.ndarray]],
    geometry: StaggeredLadderGeometryConfig,
    *,
    delta_offset_rad_us: float,
    response_probe_steps: tuple[int, ...],
    shots: int | None,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Construct static, susceptibility, curvature and reporter features."""

    rng = None if shots is None else np.random.default_rng(int(seed))
    symmetric_weights = ladder_mode_weights()[:, :3]
    static_blocks: list[np.ndarray] = []
    static_names: list[str] = []
    for probe in prepared.history_probe_steps:
        occupation = _occupation_estimate(
            prepared.history_states[int(probe)],
            prepared.precomputed,
            shots=shots,
            rng=rng,
        )
        static_blocks.append(occupation @ symmetric_weights)
        static_names.extend(
            (
                f"history_{probe}_symmetric_constant",
                f"history_{probe}_symmetric_gradient",
                f"history_{probe}_symmetric_curvature",
            )
        )

    chi_blocks: list[np.ndarray] = []
    kappa_blocks: list[np.ndarray] = []
    reporter_chi_blocks: list[np.ndarray] = []
    reporter_kappa_blocks: list[np.ndarray] = []
    chi_names: list[str] = []
    kappa_names: list[str] = []
    reporter_chi_names: list[str] = []
    reporter_kappa_names: list[str] = []
    reporter = int(geometry.defect_site)
    bulk = np.asarray([site for site in range(6) if site != reporter], dtype=int)
    epsilon = float(delta_offset_rad_us)

    for probe in response_probe_steps:
        minus = _occupation_estimate(
            branches["minus"][int(probe)],
            prepared.precomputed,
            shots=shots,
            rng=rng,
        )
        zero = _occupation_estimate(
            branches["zero"][int(probe)],
            prepared.precomputed,
            shots=shots,
            rng=rng,
        )
        plus = _occupation_estimate(
            branches["plus"][int(probe)],
            prepared.precomputed,
            shots=shots,
            rng=rng,
        )
        chi = (plus - minus) / (2.0 * epsilon)
        kappa = (plus + minus - 2.0 * zero) / (epsilon**2)
        chi_blocks.append(chi @ symmetric_weights)
        kappa_blocks.append(kappa @ symmetric_weights)
        chi_names.extend(
            (
                f"response_{probe}_chi_symmetric_constant",
                f"response_{probe}_chi_symmetric_gradient",
                f"response_{probe}_chi_symmetric_curvature",
            )
        )
        kappa_names.extend(
            (
                f"response_{probe}_kappa_symmetric_constant",
                f"response_{probe}_kappa_symmetric_gradient",
                f"response_{probe}_kappa_symmetric_curvature",
            )
        )

        chi_bulk = chi[:, bulk].mean(axis=1)
        kappa_bulk = kappa[:, bulk].mean(axis=1)
        reporter_chi_blocks.append(
            np.column_stack(
                [
                    chi[:, reporter],
                    chi[:, reporter] - chi_bulk,
                ]
            )
        )
        reporter_kappa_blocks.append(
            np.column_stack(
                [
                    kappa[:, reporter],
                    kappa[:, reporter] - kappa_bulk,
                ]
            )
        )
        reporter_chi_names.extend(
            (
                f"response_{probe}_chi_reporter",
                f"response_{probe}_chi_reporter_minus_bulk",
            )
        )
        reporter_kappa_names.extend(
            (
                f"response_{probe}_kappa_reporter",
                f"response_{probe}_kappa_reporter_minus_bulk",
            )
        )

    blocks = {
        "static_modes": np.concatenate(static_blocks, axis=1),
        "chi_modes": np.concatenate(chi_blocks, axis=1),
        "kappa_modes": np.concatenate(kappa_blocks, axis=1),
        "reporter_chi": np.concatenate(reporter_chi_blocks, axis=1),
        "reporter_kappa": np.concatenate(reporter_kappa_blocks, axis=1),
    }
    blocks["reporter_response"] = np.concatenate(
        [blocks["reporter_chi"], blocks["reporter_kappa"]],
        axis=1,
    )
    blocks["susceptibility_all"] = np.concatenate(
        [
            blocks["chi_modes"],
            blocks["kappa_modes"],
            blocks["reporter_response"],
        ],
        axis=1,
    )
    blocks["qrc_all"] = np.concatenate(
        [blocks["static_modes"], blocks["susceptibility_all"]],
        axis=1,
    )
    names = {
        "static_modes": tuple(static_names),
        "chi_modes": tuple(chi_names),
        "kappa_modes": tuple(kappa_names),
        "reporter_chi": tuple(reporter_chi_names),
        "reporter_kappa": tuple(reporter_kappa_names),
    }
    names["reporter_response"] = (
        names["reporter_chi"] + names["reporter_kappa"]
    )
    names["susceptibility_all"] = (
        names["chi_modes"]
        + names["kappa_modes"]
        + names["reporter_response"]
    )
    names["qrc_all"] = names["static_modes"] + names["susceptibility_all"]

    for key, matrix in blocks.items():
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid feature block {key}: {matrix.shape}")
        if matrix.shape[1] != len(names[key]):
            raise RuntimeError(f"feature names do not match {key}")
    return blocks, names


def classical_context_features(
    level_windows: np.ndarray,
    instability_windows: np.ndarray,
    har_path: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compact causal volatility/HAR context for the L5 warning baseline."""

    level = np.asarray(level_windows, dtype=float)
    instability = np.asarray(instability_windows, dtype=float)
    har = np.asarray(har_path, dtype=float)
    if level.ndim != 2 or instability.shape != level.shape:
        raise ValueError("level and instability windows must align")
    if har.ndim != 2 or len(har) != len(level):
        raise ValueError("HAR path must align with the input windows")

    def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
        return values[:, -min(width, values.shape[1]) :].mean(axis=1)

    def trailing_slope(values: np.ndarray, width: int) -> np.ndarray:
        local = values[:, -min(width, values.shape[1]) :]
        x = np.arange(local.shape[1], dtype=float)
        x -= x.mean()
        denominator = float(x @ x)
        if denominator <= 0:
            return np.zeros(len(values), dtype=float)
        return (local @ x) / denominator

    columns = [har]
    names = [f"har_horizon_{index + 1}" for index in range(har.shape[1])]
    for width in (5, 20, 40):
        columns.append(trailing_mean(level, width)[:, None])
        names.append(f"level_mean_{width}")
    columns.append(level[:, -1, None])
    names.append("level_last")
    for width in (5, 20, 40):
        columns.append(trailing_mean(instability, width)[:, None])
        names.append(f"instability_mean_{width}")
    columns.append(instability[:, -1, None])
    names.append("instability_last")
    for width in (5, 20):
        columns.append(trailing_slope(level, width)[:, None])
        names.append(f"level_slope_{width}")
    matrix = np.concatenate(columns, axis=1)
    return matrix, tuple(names)


def feature_families(
    classical: np.ndarray,
    classical_names: tuple[str, ...],
    qrc_blocks: dict[str, np.ndarray],
    qrc_names: dict[str, tuple[str, ...]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Return the predeclared direct and classical-plus-QRC classifiers."""

    matrices = {
        "static_qrc": qrc_blocks["static_modes"],
        "chi_qrc": qrc_blocks["chi_modes"],
        "kappa_qrc": qrc_blocks["kappa_modes"],
        "reporter_qrc": qrc_blocks["reporter_response"],
        "susceptibility_qrc": qrc_blocks["susceptibility_all"],
        "all_qrc": qrc_blocks["qrc_all"],
        "classical_plus_static": np.concatenate(
            [classical, qrc_blocks["static_modes"]], axis=1
        ),
        "classical_plus_chi": np.concatenate(
            [classical, qrc_blocks["chi_modes"]], axis=1
        ),
        "classical_plus_kappa": np.concatenate(
            [classical, qrc_blocks["kappa_modes"]], axis=1
        ),
        "classical_plus_reporter": np.concatenate(
            [classical, qrc_blocks["reporter_response"]], axis=1
        ),
        "classical_plus_susceptibility": np.concatenate(
            [classical, qrc_blocks["susceptibility_all"]], axis=1
        ),
        "classical_plus_all": np.concatenate(
            [classical, qrc_blocks["qrc_all"]], axis=1
        ),
    }
    names = {
        "static_qrc": qrc_names["static_modes"],
        "chi_qrc": qrc_names["chi_modes"],
        "kappa_qrc": qrc_names["kappa_modes"],
        "reporter_qrc": qrc_names["reporter_response"],
        "susceptibility_qrc": qrc_names["susceptibility_all"],
        "all_qrc": qrc_names["qrc_all"],
        "classical_plus_static": classical_names + qrc_names["static_modes"],
        "classical_plus_chi": classical_names + qrc_names["chi_modes"],
        "classical_plus_kappa": classical_names + qrc_names["kappa_modes"],
        "classical_plus_reporter": classical_names + qrc_names["reporter_response"],
        "classical_plus_susceptibility": (
            classical_names + qrc_names["susceptibility_all"]
        ),
        "classical_plus_all": classical_names + qrc_names["qrc_all"],
    }
    return matrices, names


def fit_warning_classifier(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    classifier_c: float,
    seed: int,
) -> np.ndarray:
    """Fit one strongly regularized linear warning head."""

    features = np.asarray(matrix, dtype=float)
    target = np.asarray(labels, dtype=int)
    train = np.asarray(train_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    if (
        features.ndim != 2
        or len(features) != len(target)
        or train.shape != (len(target),)
        or validation.shape != (len(target),)
    ):
        raise ValueError("classifier arrays are not aligned")
    if len(np.unique(target[train])) != 2:
        raise ValueError("classifier training rows require both labels")
    scaler = StandardScaler().fit(features[train])
    design = scaler.transform(features)
    model = LogisticRegression(
        C=float(classifier_c),
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=int(seed),
        max_iter=5000,
    ).fit(design[train], target[train])
    probability = np.full(len(target), np.nan, dtype=float)
    probability[validation] = model.predict_proba(design[validation])[:, 1]
    return probability


def warning_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(labels, dtype=int)[mask]
    probability = np.asarray(probabilities, dtype=float)[mask]
    if len(np.unique(target)) != 2 or not np.isfinite(probability).all():
        raise ValueError("warning metrics require finite probabilities and both labels")
    prediction = probability >= 0.5
    positives = target == 1
    negatives = ~positives
    return {
        "average_precision": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "precision_at_0p5": float(
            precision_score(target, prediction, zero_division=0)
        ),
        "recall_at_0p5": float(
            recall_score(target, prediction, zero_division=0)
        ),
        "false_positive_rate_at_0p5": float(
            np.mean(prediction[negatives]) if negatives.any() else np.nan
        ),
        "mean_probability_positive": float(probability[positives].mean()),
        "mean_probability_control": float(probability[negatives].mean()),
    }


def feature_diagnostics(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, float]:
    features = np.asarray(matrix, dtype=float)
    target = np.asarray(labels, dtype=float)
    train = np.asarray(train_mask, dtype=bool)
    correlations = []
    for column in range(features.shape[1]):
        x = features[train, column]
        if np.std(x) <= 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(x, target[train])[0, 1]))
    return {
        "feature_width": float(features.shape[1]),
        "effective_rank_train": float(effective_rank(features[train])),
        "maximum_absolute_label_correlation_train": float(
            np.max(np.abs(correlations), initial=0.0)
        ),
        "mean_feature_std_train": float(features[train].std(axis=0).mean()),
    }


def long_feature_frame(
    frame: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    names: dict[str, tuple[str, ...]],
    *,
    fold: int,
    geometry_case: str,
    estimator: str,
    shot_budget: int,
    replicate: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = frame[
        ["sample_id", "label", "episode_id", "origin_date", "fold_split"]
    ].reset_index(drop=True)
    for family, matrix in matrices.items():
        for feature_index, feature_name in enumerate(names[family]):
            rows.append(
                base.assign(
                    fold=int(fold),
                    geometry_case=geometry_case,
                    estimator=estimator,
                    shot_budget=int(shot_budget),
                    replicate=int(replicate),
                    feature_family=family,
                    feature_name=feature_name,
                    feature_value=np.asarray(matrix)[:, feature_index],
                )
            )
    return pd.concat(rows, ignore_index=True)
