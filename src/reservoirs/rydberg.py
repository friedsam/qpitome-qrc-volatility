"""True temporal Rydberg analog reservoir (exact-state simulation).

This is the historical Phase 3 implementation ported without changing its
scientific design. It uses two robust-scaled market channels, global Rydberg
controls, exact-state evolution, and occupation/pair-correlation readout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import VolatilityForecastMetrics, evaluate_volatility_forecast
from reservoirs.tfim import select_anchor_indices

Geometry = Literal["chain", "dual_chain"]
MemoryMode = Literal["temporal", "memoryless"]
OmegaMode = Literal["encode_rate", "constant"]
ObservableMode = Literal["n", "n_nn"]

C6_RAD_UM6_PER_US = 5.42e6


@dataclass(frozen=True)
class AquilaConstraints:
    """Historical public Aquila limits; re-verify before hardware work."""

    omega_max_rad_us: float = 15.8
    delta_abs_max_rad_us: float = 125.0
    total_time_max_us: float = 4.0
    min_spacing_um: float = 4.0
    area_width_um: float = 75.0
    area_height_um: float = 76.0
    omega_slew_rad_us2: float = 250.0
    delta_slew_rad_us2: float = 2500.0
    max_shots_per_task: int = 1000


@dataclass(frozen=True)
class RydbergQRCConfig:
    geometry: Geometry = "dual_chain"
    n_atoms_slow: int = 4
    n_atoms_fast: int = 4
    spacing_slow_um: float = 9.0
    spacing_fast_um: float = 15.0
    row_gap_um: float = 14.0
    chain_atoms: int = 8
    chain_spacing_um: float = 6.5

    lookback_days: int = 40
    anchor_count: int = 6
    anchor_policy: Literal["even", "recent"] = "even"
    total_time_us: float = 1.8

    delta_center_rad_us: float = 6.0
    delta_span_rad_us: float = 4.0
    omega_base_rad_us: float = 6.0
    omega_mod_frac: float = 0.5
    omega_mode: OmegaMode = "encode_rate"
    encoding: Literal["plateau", "ramp"] = "plateau"

    memory_mode: MemoryMode = "temporal"
    reverse_anchors: bool = False
    shuffle_anchors: bool = False
    shuffle_seed: int = 1234

    observable_mode: ObservableMode = "n_nn"
    collect_anchor_features: bool = True
    shots: int | None = None
    shot_seed: int = 7

    max_phase_per_substep: float = 0.25
    max_substeps_per_segment: int = 2000

    ridge_alpha: float = 10.0
    target_transform: Literal["log", "none"] = "log"
    seed: int = 42


@dataclass
class RydbergQRCResult:
    config: RydbergQRCConfig
    target: str
    train_metrics: VolatilityForecastMetrics
    val_metrics: VolatilityForecastMetrics
    test_metrics: VolatilityForecastMetrics
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    train_features: np.ndarray
    val_features: np.ndarray
    test_features: np.ndarray
    readout: Ridge
    feature_scaler: StandardScaler | None


def atom_positions(config: RydbergQRCConfig) -> np.ndarray:
    if config.geometry == "chain":
        xs = np.arange(config.chain_atoms) * config.chain_spacing_um
        return np.column_stack([xs, np.zeros(config.chain_atoms)])
    if config.geometry == "dual_chain":
        xs_slow = np.arange(config.n_atoms_slow) * config.spacing_slow_um
        xs_fast = np.arange(config.n_atoms_fast) * config.spacing_fast_um
        xs_slow = xs_slow - xs_slow.mean()
        xs_fast = xs_fast - xs_fast.mean()
        slow = np.column_stack([xs_slow, np.zeros(config.n_atoms_slow)])
        fast = np.column_stack(
            [xs_fast, np.full(config.n_atoms_fast, config.row_gap_um)]
        )
        return np.vstack([slow, fast])
    raise ValueError(f"Unknown geometry: {config.geometry}")


def interaction_matrix(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=-1))
    with np.errstate(divide="ignore"):
        interactions = C6_RAD_UM6_PER_US / distances**6
    np.fill_diagonal(interactions, 0.0)
    return interactions


@dataclass
class _Precomputed:
    n_atoms: int
    positions: np.ndarray
    v_matrix: np.ndarray
    occ_bits: np.ndarray
    e_int: np.ndarray
    n_occ: np.ndarray
    pair_index: list[tuple[int, int]]
    pair_bits: np.ndarray


def precompute(config: RydbergQRCConfig) -> _Precomputed:
    positions = atom_positions(config)
    n_atoms = positions.shape[0]
    if n_atoms > 14:
        raise ValueError(f"{n_atoms} atoms exceeds exact-state practical limit (14)")
    v_matrix = interaction_matrix(positions)
    dimension = 2**n_atoms
    indices = np.arange(dimension)
    occupations = np.stack(
        [((indices >> (n_atoms - 1 - qubit)) & 1) for qubit in range(n_atoms)],
        axis=1,
    ).astype(float)
    interaction_energy = np.einsum(
        "si,ij,sj->s", occupations, np.triu(v_matrix, k=1), occupations
    )
    total_occupation = occupations.sum(axis=1)
    pair_index = [
        (left, right)
        for left in range(n_atoms - 1)
        for right in range(left + 1, n_atoms)
    ]
    pair_bits = (
        np.stack(
            [occupations[:, left] * occupations[:, right] for left, right in pair_index],
            axis=1,
        )
        if pair_index
        else np.zeros((dimension, 0))
    )
    return _Precomputed(
        n_atoms=n_atoms,
        positions=positions,
        v_matrix=v_matrix,
        occ_bits=occupations,
        e_int=interaction_energy,
        n_occ=total_occupation,
        pair_index=pair_index,
        pair_bits=pair_bits,
    )


def _apply_global_rx_batch(
    states: np.ndarray,
    thetas: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    cosines = np.cos(thetas / 2.0)
    sines = np.sin(thetas / 2.0)
    for qubit in range(n_atoms):
        left = 2**qubit
        right = 2 ** (n_atoms - 1 - qubit)
        reshaped = states.reshape(-1, left, 2, right)
        amplitude_zero = reshaped[:, :, 0, :].copy()
        amplitude_one = reshaped[:, :, 1, :]
        cosine = cosines[:, None, None]
        sine = sines[:, None, None]
        reshaped[:, :, 0, :] = cosine * amplitude_zero - 1j * sine * amplitude_one
        reshaped[:, :, 1, :] = -1j * sine * amplitude_zero + cosine * amplitude_one
        states = reshaped.reshape(-1, 2**n_atoms)
    return states


def _evolve_segment_batch(
    states: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
    segment_time: float,
    precomputed: _Precomputed,
    config: RydbergQRCConfig,
) -> np.ndarray:
    interaction_scale = float(precomputed.e_int.max(initial=0.0))
    scale = max(
        np.abs(delta).max(initial=0.0),
        np.abs(omega).max(initial=0.0),
        interaction_scale,
        1e-9,
    )
    substeps = int(np.ceil(scale * segment_time / config.max_phase_per_substep))
    substeps = int(np.clip(substeps, 1, config.max_substeps_per_segment))
    dt = segment_time / substeps
    diagonal = precomputed.e_int[None, :] - delta[:, None] * precomputed.n_occ[None, :]
    half_phase = np.exp(-0.5j * dt * diagonal)
    full_phase = half_phase * half_phase
    thetas = omega * dt
    states = states * half_phase
    for step in range(substeps):
        states = _apply_global_rx_batch(states, thetas, precomputed.n_atoms)
        states = states * (half_phase if step == substeps - 1 else full_phase)
    return states


def _evolve_ramp_segment_batch(
    states: np.ndarray,
    omega: np.ndarray,
    delta_start: np.ndarray,
    delta_end: np.ndarray,
    segment_time: float,
    precomputed: _Precomputed,
    config: RydbergQRCConfig,
) -> np.ndarray:
    interaction_scale = float(precomputed.e_int.max(initial=0.0))
    delta_scale = max(
        np.abs(delta_start).max(initial=0.0),
        np.abs(delta_end).max(initial=0.0),
    )
    scale = max(
        delta_scale,
        np.abs(omega).max(initial=0.0),
        interaction_scale,
        1e-9,
    )
    substeps = int(np.ceil(scale * segment_time / config.max_phase_per_substep))
    substeps = int(np.clip(substeps, 1, config.max_substeps_per_segment))
    dt = segment_time / substeps
    thetas = omega * dt
    slope = (delta_end - delta_start) / segment_time
    for step in range(substeps):
        midpoint = (step + 0.5) * dt
        delta_midpoint = delta_start + slope * midpoint
        diagonal = (
            precomputed.e_int[None, :]
            - delta_midpoint[:, None] * precomputed.n_occ[None, :]
        )
        half_phase = np.exp(-0.5j * dt * diagonal)
        states = states * half_phase
        states = _apply_global_rx_batch(states, thetas, precomputed.n_atoms)
        states = states * half_phase
    return states


def _drive_from_window(
    windows: np.ndarray,
    anchor_indices: np.ndarray,
    config: RydbergQRCConfig,
    constraints: AquilaConstraints,
) -> tuple[np.ndarray, np.ndarray]:
    level = windows[:, anchor_indices, 0]
    rate = windows[:, anchor_indices, 1]
    deltas = config.delta_center_rad_us + config.delta_span_rad_us * level
    deltas = np.clip(
        deltas,
        -constraints.delta_abs_max_rad_us,
        constraints.delta_abs_max_rad_us,
    )
    if config.omega_mode == "encode_rate":
        omegas = config.omega_base_rad_us * (1.0 + config.omega_mod_frac * rate)
    elif config.omega_mode == "constant":
        omegas = np.full_like(deltas, config.omega_base_rad_us)
    else:
        raise ValueError(f"Unknown omega_mode: {config.omega_mode}")
    omegas = np.clip(omegas, 0.0, constraints.omega_max_rad_us)
    return deltas, omegas


def _measure_features(
    states: np.ndarray,
    precomputed: _Precomputed,
    config: RydbergQRCConfig,
    random_generator: np.random.Generator | None,
) -> np.ndarray:
    probabilities = np.abs(states) ** 2
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    if config.shots is not None:
        if random_generator is None:
            raise ValueError("random_generator is required when shots is set")
        counts = random_generator.multinomial(config.shots, probabilities)
        probabilities = counts / float(config.shots)
    features = [probabilities @ precomputed.occ_bits]
    if config.observable_mode == "n_nn":
        features.append(probabilities @ precomputed.pair_bits)
    return np.concatenate(features, axis=1)


def select_rydberg_anchor_indices(config: RydbergQRCConfig) -> np.ndarray:
    anchor_indices = select_anchor_indices(
        config.lookback_days,
        config.anchor_count,
        config.anchor_policy,
    )
    if config.reverse_anchors:
        anchor_indices = anchor_indices[::-1]
    if config.shuffle_anchors:
        random_generator = np.random.default_rng(config.shuffle_seed)
        anchor_indices = anchor_indices[
            random_generator.permutation(len(anchor_indices))
        ]
    return np.asarray(anchor_indices, dtype=int)


def build_rydberg_feature_matrix(
    X_windows: np.ndarray,
    config: RydbergQRCConfig,
    *,
    hw: AquilaConstraints | None = None,
    verbose: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if X_windows.ndim != 3 or X_windows.shape[2] != 2:
        raise ValueError(f"Expected windows (samples, lookback, 2), got {X_windows.shape}")
    constraints = hw or AquilaConstraints()
    precomputed = precompute(config)
    sample_count = X_windows.shape[0]
    dimension = 2**precomputed.n_atoms
    anchor_indices = select_rydberg_anchor_indices(config)
    anchor_count = len(anchor_indices)
    segment_time = config.total_time_us / anchor_count
    deltas, omegas = _drive_from_window(
        X_windows,
        anchor_indices,
        config,
        constraints,
    )
    shot_rng = rng
    if config.shots is not None and shot_rng is None:
        shot_rng = np.random.default_rng(config.shot_seed)

    def fresh_states() -> np.ndarray:
        states = np.zeros((sample_count, dimension), dtype=complex)
        states[:, 0] = 1.0
        return states

    per_anchor: list[np.ndarray] = []
    use_ramp = config.encoding == "ramp"
    if config.memory_mode == "temporal":
        states = fresh_states()
        for anchor in range(anchor_count):
            if verbose:
                print(f"Rydberg segment {anchor + 1}/{anchor_count} (batch {sample_count})")
            if use_ramp:
                delta_start = deltas[:, anchor - 1] if anchor > 0 else deltas[:, 0]
                states = _evolve_ramp_segment_batch(
                    states,
                    omegas[:, anchor],
                    delta_start,
                    deltas[:, anchor],
                    segment_time,
                    precomputed,
                    config,
                )
            else:
                states = _evolve_segment_batch(
                    states,
                    omegas[:, anchor],
                    deltas[:, anchor],
                    segment_time,
                    precomputed,
                    config,
                )
            if config.collect_anchor_features or anchor == anchor_count - 1:
                per_anchor.append(
                    _measure_features(states, precomputed, config, shot_rng)
                )
    elif config.memory_mode == "memoryless":
        for anchor in range(anchor_count):
            if verbose:
                print(
                    f"Rydberg memoryless segment {anchor + 1}/{anchor_count} "
                    f"(batch {sample_count})"
                )
            if use_ramp:
                delta_start = deltas[:, anchor - 1] if anchor > 0 else deltas[:, 0]
                states = _evolve_ramp_segment_batch(
                    fresh_states(),
                    omegas[:, anchor],
                    delta_start,
                    deltas[:, anchor],
                    segment_time,
                    precomputed,
                    config,
                )
            else:
                states = _evolve_segment_batch(
                    fresh_states(),
                    omegas[:, anchor],
                    deltas[:, anchor],
                    segment_time,
                    precomputed,
                    config,
                )
            per_anchor.append(_measure_features(states, precomputed, config, shot_rng))
    else:
        raise ValueError(f"Unknown memory_mode: {config.memory_mode}")
    return np.concatenate(per_anchor, axis=1)


def ensure_market_scalar(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    output = frame.copy()
    if column in output.columns:
        return output
    if column == "rv_accel_log_5_20":
        output[column] = np.log(
            output["rv_5d"].clip(lower=1e-12)
            / output["rv_20d"].clip(lower=1e-12)
        )
    elif column == "vix_rv_spread":
        output[column] = np.log(
            (output["vix_close"].clip(lower=1e-12) / 100.0)
            / output["rv_20d"].clip(lower=1e-12)
        )
    else:
        raise ValueError(f"Unknown scalar column {column!r}")
    return output


def _robust_scale_params(
    train_values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    lower = float(np.nanquantile(train_values, lower_quantile))
    upper = float(np.nanquantile(train_values, upper_quantile))
    median = float(np.nanmedian(train_values))
    half_range = 0.5 * (upper - lower)
    if not np.isfinite(half_range) or half_range <= 0:
        raise ValueError("Invalid robust scale")
    return median, half_range


def make_level_rate_sequence_splits(
    splits: dict[str, pd.DataFrame],
    *,
    level_col: str,
    rate_col: str,
    target_column: str,
    lookback_days: int,
    clip_q_low: float = 0.01,
    clip_q_high: float = 0.99,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.Series]]:
    prepared = {
        name: ensure_market_scalar(ensure_market_scalar(frame, level_col), rate_col)
        for name, frame in splits.items()
    }
    train = prepared["train"]
    parameters = {
        column: _robust_scale_params(
            train[column].to_numpy(float),
            clip_q_low,
            clip_q_high,
        )
        for column in (level_col, rate_col)
    }
    output: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]] = {}
    for name, frame in prepared.items():
        channels = []
        for column in (level_col, rate_col):
            median, half_range = parameters[column]
            channels.append(
                np.clip(
                    (frame[column].to_numpy(float) - median) / half_range,
                    -1.0,
                    1.0,
                )
            )
        values = np.column_stack(channels)
        targets = frame[target_column].to_numpy(float)
        dates = frame["date"].reset_index(drop=True)
        windows: list[np.ndarray] = []
        labels: list[float] = []
        label_dates: list[object] = []
        for end in range(lookback_days - 1, len(frame)):
            windows.append(values[end - lookback_days + 1 : end + 1])
            labels.append(targets[end])
            label_dates.append(dates.iloc[end])
        output[name] = (
            np.asarray(windows, float),
            np.asarray(labels, float),
            pd.Series(label_dates),
        )
    return output


def fit_qrc_readout(
    H_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: RydbergQRCConfig,
) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(H_train)
    readout = Ridge(alpha=config.ridge_alpha)
    if config.target_transform == "log":
        target = np.log(np.maximum(y_train, 1e-8))
    elif config.target_transform == "none":
        target = y_train
    else:
        raise ValueError(f"Unknown target_transform: {config.target_transform}")
    readout.fit(transformed, target)
    return readout, scaler


def predict_qrc_readout(
    readout: Ridge,
    scaler: StandardScaler,
    features: np.ndarray,
    *,
    config: RydbergQRCConfig,
) -> np.ndarray:
    raw = readout.predict(scaler.transform(features))
    if config.target_transform == "log":
        return np.exp(raw)
    return np.maximum(raw, 1e-8)


def fit_rydberg_qrc_regressor(
    sequence_splits: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]],
    *,
    config: RydbergQRCConfig,
    target: str,
    verbose: bool = False,
) -> RydbergQRCResult:
    X_train, y_train, _ = sequence_splits["train"]
    X_val, y_val, _ = sequence_splits["val"]
    X_test, y_test, _ = sequence_splits["test"]
    H_train = build_rydberg_feature_matrix(X_train, config, verbose=verbose)
    H_val = build_rydberg_feature_matrix(X_val, config, verbose=verbose)
    H_test = build_rydberg_feature_matrix(X_test, config, verbose=verbose)
    readout, scaler = fit_qrc_readout(H_train, y_train, config=config)
    train_predictions = predict_qrc_readout(readout, scaler, H_train, config=config)
    val_predictions = predict_qrc_readout(readout, scaler, H_val, config=config)
    test_predictions = predict_qrc_readout(readout, scaler, H_test, config=config)
    return RydbergQRCResult(
        config=config,
        target=target,
        train_metrics=evaluate_volatility_forecast(y_train, train_predictions),
        val_metrics=evaluate_volatility_forecast(y_val, val_predictions),
        test_metrics=evaluate_volatility_forecast(y_test, test_predictions),
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        train_features=H_train,
        val_features=H_val,
        test_features=H_test,
        readout=readout,
        feature_scaler=scaler,
    )


def summarize_rydberg_result(result: RydbergQRCResult) -> dict:
    row = asdict(result.config)
    row.update(
        {
            "model": "rydberg_temporal_reservoir_exact",
            "target": result.target,
            "train_n": len(result.train_predictions),
            "val_n": len(result.val_predictions),
            "test_n": len(result.test_predictions),
            "n_reservoir_features": result.train_features.shape[1],
        }
    )
    for split, metrics in (
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
    ):
        row[f"{split}_rmse"] = metrics.rmse
        row[f"{split}_qlike"] = metrics.qlike
        row[f"{split}_mz_alpha"] = metrics.mz_alpha
        row[f"{split}_mz_beta"] = metrics.mz_beta
        row[f"{split}_mz_r2"] = metrics.mz_r2
    return row


def validate_aquila_feasibility(
    config: RydbergQRCConfig,
    hw: AquilaConstraints | None = None,
) -> dict:
    constraints = hw or AquilaConstraints()
    precomputed = precompute(config)
    positions = precomputed.positions
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt((differences**2).sum(-1))
    np.fill_diagonal(distances, np.inf)
    minimum_spacing = float(distances.min())
    width = float(positions[:, 0].max() - positions[:, 0].min())
    height = float(positions[:, 1].max() - positions[:, 1].min())
    delta_low = config.delta_center_rad_us - abs(config.delta_span_rad_us)
    delta_high = config.delta_center_rad_us + abs(config.delta_span_rad_us)
    omega_high = config.omega_base_rad_us * (1.0 + abs(config.omega_mod_frac))
    omega_low = max(
        config.omega_base_rad_us * (1.0 - abs(config.omega_mod_frac)),
        0.0,
    )
    omega_ramp = (omega_high - omega_low) / constraints.omega_slew_rad_us2
    delta_ramp = (delta_high - delta_low) / constraints.delta_slew_rad_us2
    ramp_total = (config.anchor_count - 1) * max(omega_ramp, delta_ramp)
    checks = {
        "min_spacing_ok": minimum_spacing >= constraints.min_spacing_um,
        "area_ok": width <= constraints.area_width_um
        and height <= constraints.area_height_um,
        "omega_range_ok": omega_high <= constraints.omega_max_rad_us,
        "delta_range_ok": max(abs(delta_low), abs(delta_high))
        <= constraints.delta_abs_max_rad_us,
        "total_time_ok": config.total_time_us <= constraints.total_time_max_us,
        "time_with_ramps_ok": config.total_time_us + ramp_total
        <= constraints.total_time_max_us,
        "shots_ok": (config.shots or 0) <= constraints.max_shots_per_task,
    }
    return {
        "feasible": all(checks.values()),
        "checks": checks,
        "min_spacing_um": minimum_spacing,
        "array_width_um": width,
        "array_height_um": height,
        "delta_range_rad_us": (delta_low, delta_high),
        "omega_range_rad_us": (omega_low, omega_high),
        "estimated_ramp_overhead_us": ramp_total,
        "n_atoms": precomputed.n_atoms,
        "nn_interaction_slow_rad_us": float(
            C6_RAD_UM6_PER_US / config.spacing_slow_um**6
        )
        if config.geometry == "dual_chain"
        else float(C6_RAD_UM6_PER_US / config.chain_spacing_um**6),
        "verify_specs_note": "Re-verify Aquila limits against current QuEra/Braket docs.",
    }
