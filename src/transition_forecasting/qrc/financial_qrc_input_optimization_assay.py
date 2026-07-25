from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.bivariate_carrier_crossmix_assay import (
    CARRIER_MASKS,
    evolve_carrier_mask_probabilities,
)
from transition_forecasting.qrc.financial_qrc_feature_transfer_assay import (
    _aggregate_fold_metrics,
    _cell_frame,
    _fold_metric_rows,
    _label_gap_rows,
)
from transition_forecasting.qrc.input_admission_assay import (
    load_ohlc_panel,
    parse_ticker,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    probabilities_to_occupations,
)
from transition_forecasting.qrc.ladder_mode_readout_tools import (
    ladder_mode_weights,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    build_candidate_sequences,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    effective_rank,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    _select_rows_for_fold,
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

INPUT_REPRESENTATIONS = (
    "level_instability",
    "level_signed_return",
    "level_signed_range",
)
MODE_BANKS = (
    "six_density_curvature",
    "nine_density_curvature_antisymmetric_curvature",
)


@dataclass(frozen=True)
class FinancialQRCInputOptimizationConfig:
    """Bounded task-specific optimization around the successful incumbent QRC."""

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    lead: int = 5
    max_per_class: int = 12
    maximum_window: int = 40
    windows: tuple[int, ...] = (20, 40)
    prequential_blocks: int = 5
    ridge_alpha: float = 100.0
    min_causal_residual_rows: int = 10
    selection_seed: int = 20260721
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    split_horizon: int = 4
    step_duration_us: float = 0.03
    interaction_scale: float = 1.25
    probe_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
    input_representations: tuple[str, ...] = INPUT_REPRESENTATIONS
    mode_banks: tuple[str, ...] = MODE_BANKS

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if self.lead != 5:
            raise ValueError("input optimization is intentionally restricted to L5")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.maximum_window < 5:
            raise ValueError("maximum_window must be at least five")
        if not self.windows or any(
            window < 5 or window > self.maximum_window for window in self.windows
        ):
            raise ValueError("windows must lie in [5, maximum_window]")
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("windows must be unique")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.min_causal_residual_rows < 5:
            raise ValueError("min_causal_residual_rows must be at least five")
        if self.step_duration_us <= 0:
            raise ValueError("step_duration_us must be positive")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if set(self.input_representations).difference(INPUT_REPRESENTATIONS):
            raise ValueError("unsupported input representation")
        if set(self.mode_banks).difference(MODE_BANKS):
            raise ValueError("unsupported compact mode bank")
        if not self.probe_fractions:
            raise ValueError("probe_fractions cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def signed_market_channels(
    market: pd.DataFrame,
    origin_date: pd.Timestamp,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return bidirectional stress-return and signed-range channels.

    Negative close-to-close returns map to positive stress. Intraday range is
    assigned the same stress sign, so both channels use both sides of the
    amplitude-control interval after train-only robust centering/scaling.
    """

    local = market.loc[:origin_date].dropna(subset=["close"]).tail(window + 1)
    if len(local) != window + 1:
        return None
    values = local[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return None
    _, high, low, close = values.T
    if np.any(high < low):
        return None
    close_return = np.diff(np.log(close))
    intraday_range = np.log(high[1:] / low[1:])
    stress_return = -close_return
    signed_range = -np.sign(close_return) * intraday_range
    if not (
        len(stress_return) == len(signed_range) == window
        and np.isfinite(stress_return).all()
        and np.isfinite(signed_range).all()
    ):
        return None
    return stress_return, signed_range


def compact_mode_banks(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Return the historical six modes and one restrained nine-mode extension."""

    occupation = probabilities_to_occupations(probabilities)
    weights = ladder_mode_weights()
    modes = np.einsum("rpi,ik->rpk", occupation, weights)
    six = modes[:, :, (0, 2)].reshape(len(modes), -1)
    nine = modes[:, :, (0, 2, 5)].reshape(len(modes), -1)
    banks = {
        "six_density_curvature": six,
        "nine_density_curvature_antisymmetric_curvature": nine,
    }
    if six.shape[1] != probabilities.shape[1] * 2:
        raise RuntimeError("unexpected six-mode width")
    if nine.shape[1] != probabilities.shape[1] * 3:
        raise RuntimeError("unexpected nine-mode width")
    if any(not np.isfinite(matrix).all() for matrix in banks.values()):
        raise RuntimeError("compact mode extraction produced non-finite values")
    return banks


def fit_fixed_signed_probe(
    features: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    *,
    ridge_alpha: float,
    pca_components: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one fixed-alpha, no-intercept multi-output residual probe."""

    matrix = np.asarray(features, dtype=float)
    targets = np.asarray(residuals, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if matrix.ndim != 2 or targets.ndim != 2 or len(matrix) != len(targets):
        raise ValueError("features and residuals must be aligned matrices")
    if fit.shape != (len(matrix),) or not fit.any():
        raise ValueError("fit_mask must select aligned rows")
    if not np.isfinite(matrix).all() or not np.isfinite(targets[fit]).all():
        raise ValueError("probe inputs must be finite on fit rows")

    scaler = StandardScaler().fit(matrix[fit])
    transformed = scaler.transform(matrix)
    explained = 1.0
    if pca_components is not None:
        maximum = min(int(fit.sum()), transformed.shape[1])
        if not 1 <= int(pca_components) <= maximum:
            raise ValueError("pca_components exceed the available fit rank")
        pca = PCA(n_components=int(pca_components), svd_solver="full").fit(
            transformed[fit]
        )
        transformed = pca.transform(transformed)
        explained = float(pca.explained_variance_ratio_.sum())

    model = Ridge(alpha=float(ridge_alpha), fit_intercept=False).fit(
        transformed[fit], targets[fit]
    )
    correction = np.asarray(model.predict(transformed), dtype=float)
    intercept = float(np.max(np.abs(np.atleast_1d(model.intercept_))))
    if intercept > 0.0:
        raise RuntimeError("fixed no-intercept probe acquired a nonzero intercept")
    return correction, {
        "ridge_alpha": float(ridge_alpha),
        "fit_rows": int(fit.sum()),
        "feature_width": int(matrix.shape[1]),
        "transformed_width": int(transformed.shape[1]),
        "pca_components": None if pca_components is None else int(pca_components),
        "pca_explained_variance": explained,
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "intercept_max_abs": intercept,
    }


def _prepare_fold(
    *,
    dataset: object,
    panel: dict[str, pd.DataFrame],
    level_channel: int | None,
    fold: int,
    config: FinancialQRCInputOptimizationConfig,
    candidate_features: CandidateFeatureConfig,
) -> dict[str, object]:
    selected = _select_rows_for_fold(
        dataset.manifest,
        fold=int(fold),
        leads=(int(config.lead),),
        max_per_class=int(config.max_per_class),
        excluded_ids=set(),
        seed=int(config.selection_seed),
    )
    tensor_rows = selected["_tensor_row"].to_numpy(dtype=int)
    level = extract_level_windows(
        dataset,
        tensor_rows,
        sequence_length=config.maximum_window,
        level_channel=level_channel,
    )

    stress_return = np.full_like(level, np.nan, dtype=float)
    signed_range = np.full_like(level, np.nan, dtype=float)
    market_valid = np.zeros(len(selected), dtype=bool)
    parsed_dates = pd.to_datetime(
        selected["origin_date"], format="mixed", utc=True
    ).dt.normalize()
    tickers = selected["sample_id"].astype(str).map(parse_ticker)
    for row, (ticker, date) in enumerate(zip(tickers, parsed_dates)):
        if ticker not in panel:
            continue
        extracted = signed_market_channels(
            panel[ticker], pd.Timestamp(date), window=config.maximum_window
        )
        if extracted is None:
            continue
        stress_return[row], signed_range[row] = extracted
        market_valid[row] = True

    usable = (
        dataset.valid[tensor_rows]
        & np.isfinite(level).all(axis=1)
        & market_valid
        & np.isfinite(stress_return).all(axis=1)
        & np.isfinite(signed_range).all(axis=1)
    )
    frame = selected.loc[usable].reset_index(drop=True)
    level = level[usable]
    stress_return = stress_return[usable]
    signed_range = signed_range[usable]
    tensor_rows = tensor_rows[usable]
    if frame.empty or frame["fold_split"].eq("test").any():
        raise RuntimeError(f"fold {fold}: invalid common-row selection")

    train = frame["fold_split"].eq("train").to_numpy()
    validation = frame["fold_split"].eq("val").to_numpy()
    if not train.any() or not validation.any():
        raise RuntimeError(f"fold {fold}: empty train or validation split")

    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _fit_har(frame, y, train)
    residuals, residual_train_mask = _prequential_har_residuals(
        frame, y, train, blocks=config.prequential_blocks
    )
    if int(residual_train_mask.sum()) < config.min_causal_residual_rows:
        raise RuntimeError(
            f"fold {fold}: only {int(residual_train_mask.sum())} causal residual rows"
        )

    encoded: dict[tuple[str, int], np.ndarray] = {}
    scaler_rows: list[dict[str, object]] = []
    for window in config.windows:
        level_window = level[:, -int(window) :]
        channels = {
            "level_instability": build_candidate_sequences(
                level_window, "level_instability", candidate_features
            ),
            "level_signed_return": np.stack(
                [level_window, stress_return[:, -int(window) :]], axis=-1
            ),
            "level_signed_range": np.stack(
                [level_window, signed_range[:, -int(window) :]], axis=-1
            ),
        }
        for representation in config.input_representations:
            raw = channels[representation]
            scaler = fit_channel_scaler(raw, train)
            transformed = transform_candidate_sequences(raw, scaler)
            encoded[(representation, int(window))] = transformed
            scaler_rows.append(
                {
                    "fold": int(fold),
                    "representation": representation,
                    "window": int(window),
                    "train_rows": int(train.sum()),
                    "validation_rows": int(validation.sum()),
                    "causal_residual_rows": int(residual_train_mask.sum()),
                    "channel_0_median": float(scaler.medians[0]),
                    "channel_1_median": float(scaler.medians[1]),
                    "channel_0_half_range": float(scaler.half_ranges[0]),
                    "channel_1_half_range": float(scaler.half_ranges[1]),
                    "channel_0_clip_fraction": float(
                        np.mean(np.isclose(np.abs(transformed[:, :, 0]), 1.0))
                    ),
                    "channel_1_clip_fraction": float(
                        np.mean(np.isclose(np.abs(transformed[:, :, 1]), 1.0))
                    ),
                    "channel_1_zero_fraction": float(
                        np.mean(np.isclose(transformed[:, :, 1], 0.0, atol=1e-14))
                    ),
                }
            )

    return {
        "fold": int(fold),
        "frame": frame,
        "tensor_rows": tensor_rows,
        "train": train,
        "validation": validation,
        "y": y,
        "har": har,
        "residuals": residuals,
        "residual_train_mask": residual_train_mask,
        "encoded": encoded,
        "scaler_rows": scaler_rows,
    }


def _shuffle_sequences(values: np.ndarray, *, seed: int) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    rng = np.random.default_rng(int(seed))
    for row in range(len(result)):
        result[row] = result[row, rng.permutation(result.shape[1])]
    return result


def _architecture_name(
    representation: str,
    window: int,
    bank: str,
    *,
    condition: str = "interaction_on",
) -> str:
    return f"{representation}__w{window}__{bank}__{condition}"


def run_financial_qrc_input_optimization_assay(
    *,
    fold_dir: Path,
    panel_path: Path,
    results_root: Path,
    config: FinancialQRCInputOptimizationConfig = FinancialQRCInputOptimizationConfig(),
    candidate_features: CandidateFeatureConfig = CandidateFeatureConfig(),
    reservoir: TemporalRydbergChainConfig | None = None,
    geometry: StaggeredLadderGeometryConfig | None = None,
    run_id: str | None = None,
) -> Path:
    """Test task-specific inputs without reopening reservoir-parameter tuning."""

    config.validate()
    candidate_features.validate()
    dataset = load_rolling_fold_dataset(Path(fold_dir))
    panel = load_ohlc_panel(Path(panel_path))
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )
    base_reservoir = reservoir or TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=config.step_duration_us,
        probe_fractions=config.probe_fractions,
        shots=None,
        shot_seed=config.selection_seed,
    )
    base_reservoir.validate()
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    geometry_config.validate()
    if base_reservoir.n_atoms != 6 or base_reservoir.shots is not None:
        raise ValueError("input optimization requires exact six-atom simulation")

    prepared = [
        _prepare_fold(
            dataset=dataset,
            panel=panel,
            level_channel=level_channel,
            fold=int(fold),
            config=config,
            candidate_features=candidate_features,
        )
        for fold in config.folds
    ]

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "panel_path": str(panel_path),
            "config": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "test_rows_used": 0,
            "fit_intercept": False,
            "ridge_selection": "fixed_predeclared_alpha",
            "lambda_calibration": False,
            "candidate_set": (
                "three predeclared inputs, two predeclared history lengths, "
                "two compact mode banks"
            ),
            "artifact_guard": (
                "No winner-specific tuning. Promotion requires fold-consistent directional "
                "and metric gains, superiority to a matched raw-input PCA control, and "
                "failure of interaction-off or temporal-shuffle controls."
            ),
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=False)

    cell_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    scaler_rows: list[dict[str, object]] = []
    retained_rows: list[pd.DataFrame] = []

    def evaluate(
        payload: dict[str, object],
        *,
        architecture: str,
        matrix: np.ndarray,
        pca_components: int | None = None,
    ) -> None:
        correction, diagnostics = fit_fixed_signed_probe(
            matrix,
            payload["residuals"],
            payload["residual_train_mask"],
            ridge_alpha=config.ridge_alpha,
            pca_components=pca_components,
        )
        fit_rows.append(
            {"fold": int(payload["fold"]), "architecture": architecture, **diagnostics}
        )
        train_matrix = np.asarray(matrix[payload["train"]], dtype=float)
        feature_rows.append(
            {
                "fold": int(payload["fold"]),
                "architecture": architecture,
                "rows": int(len(matrix)),
                "features": int(matrix.shape[1]),
                "effective_rank_train": float(effective_rank(train_matrix)),
                "numerical_rank_train": int(
                    np.linalg.matrix_rank(
                        train_matrix - train_matrix.mean(axis=0, keepdims=True)
                    )
                ),
            }
        )
        cells = _cell_frame(
            payload["frame"],
            payload["y"],
            payload["har"],
            correction,
            payload["validation"],
            fold=int(payload["fold"]),
            architecture=architecture,
            split_horizon=config.split_horizon,
        )
        cell_frames.append(cells)
        fold_rows.extend(_fold_metric_rows(cells))
        gap_rows.extend(_label_gap_rows(cells))

    for payload in prepared:
        fold = int(payload["fold"])
        scaler_rows.extend(payload["scaler_rows"])
        for representation in config.input_representations:
            for window in config.windows:
                encoded = payload["encoded"][(representation, int(window))]
                local_reservoir = replace(
                    base_reservoir,
                    step_duration_us=config.step_duration_us,
                    probe_fractions=config.probe_fractions,
                    shot_seed=config.selection_seed + 1000 * fold + int(window),
                )
                probabilities, metadata = evolve_carrier_mask_probabilities(
                    encoded,
                    local_reservoir,
                    geometry_config,
                    np.asarray(CARRIER_MASKS["identity"], dtype=float),
                    mask_name="identity_incumbent",
                    interaction_scale=config.interaction_scale,
                    drive_phase_rad=0.0,
                )
                banks = compact_mode_banks(probabilities)
                np.savez_compressed(
                    probability_dir
                    / f"fold_{fold}__{representation}__w{window}__interaction_on.npz",
                    encoded_sequences=encoded,
                    probabilities=probabilities,
                    sample_id=payload["frame"]["sample_id"].astype(str).to_numpy(),
                    fold_split=payload["frame"]["fold_split"].astype(str).to_numpy(),
                    label=payload["frame"]["label"].to_numpy(dtype=int),
                    target_path=payload["y"],
                    har_prediction_path=payload["har"],
                    prequential_residual_path=payload["residuals"],
                    prequential_residual_valid=payload["residual_train_mask"],
                )
                (run_dir / f"fold_{fold}__{representation}__w{window}_metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                for bank in config.mode_banks:
                    evaluate(
                        payload,
                        architecture=_architecture_name(
                            representation, int(window), bank
                        ),
                        matrix=banks[bank],
                    )
                    evaluate(
                        payload,
                        architecture=(
                            f"{representation}__w{window}__raw_pca"
                            f"{banks[bank].shape[1]}__classical"
                        ),
                        matrix=encoded.reshape(len(encoded), -1),
                        pca_components=banks[bank].shape[1],
                    )

        reference = payload["encoded"][("level_instability", config.maximum_window)]
        for condition, sequences, interaction_scale in (
            (
                "interaction_off",
                reference,
                0.0,
            ),
            (
                "temporal_shuffle",
                _shuffle_sequences(
                    reference,
                    seed=config.selection_seed + 100_003 * fold,
                ),
                config.interaction_scale,
            ),
        ):
            probabilities, metadata = evolve_carrier_mask_probabilities(
                sequences,
                replace(base_reservoir, shot_seed=config.selection_seed + 50_000 + fold),
                geometry_config,
                np.asarray(CARRIER_MASKS["identity"], dtype=float),
                mask_name=f"identity_{condition}",
                interaction_scale=float(interaction_scale),
                drive_phase_rad=0.0,
            )
            banks = compact_mode_banks(probabilities)
            for bank in config.mode_banks:
                evaluate(
                    payload,
                    architecture=_architecture_name(
                        "level_instability",
                        config.maximum_window,
                        bank,
                        condition=condition,
                    ),
                    matrix=banks[bank],
                )

        retained = payload["frame"][
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
        retained["tensor_row"] = payload["tensor_rows"]
        retained_rows.append(retained)

    cells = pd.concat(cell_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    label_gaps = pd.DataFrame(gap_rows)
    architecture_summary = _aggregate_fold_metrics(fold_metrics)
    label_gap_summary = (
        label_gaps.groupby(["architecture", "segment"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            positive_gap_folds=(
                "transition_minus_control_correction",
                lambda values: int((values > 0.0).sum()),
            ),
            mean_fold_transition_minus_control=(
                "transition_minus_control_correction",
                "mean",
            ),
            minimum_fold_transition_minus_control=(
                "transition_minus_control_correction",
                "min",
            ),
        )
        .sort_values(["architecture", "segment"])
    )

    all_rows = architecture_summary.loc[
        architecture_summary["population"].eq("all")
        & architecture_summary["segment"].eq("all")
    ].copy()
    all_rows["promotion_gate"] = (
        all_rows["positive_correlation_folds"].ge(4)
        & all_rows["positive_residual_sign_gap_folds"].ge(4)
        & (
            all_rows["qlike_fold_wins"].ge(4)
            | all_rows["rmse_fold_wins"].ge(4)
        )
        & all_rows["mean_fold_rmse_delta"].le(0.0)
    )
    all_rows = all_rows.sort_values(
        [
            "promotion_gate",
            "rmse_fold_wins",
            "qlike_fold_wins",
            "mean_fold_rmse_delta",
            "mean_fold_qlike_delta",
        ],
        ascending=[False, False, False, True, True],
    )

    cells.to_csv(run_dir / "validation_cells.csv.gz", index=False, compression="gzip")
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    architecture_summary.to_csv(run_dir / "architecture_summary.csv", index=False)
    all_rows.to_csv(run_dir / "primary_comparison.csv", index=False)
    label_gaps.to_csv(run_dir / "label_gap_by_fold.csv", index=False)
    label_gap_summary.to_csv(run_dir / "label_gap_summary.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(run_dir / "readout_fits.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(run_dir / "feature_diagnostics.csv", index=False)
    pd.DataFrame(scaler_rows).to_csv(run_dir / "input_scalers.csv", index=False)
    pd.concat(retained_rows, ignore_index=True).to_csv(
        run_dir / "selected_samples.csv", index=False
    )

    summary = {
        "status": "financial_qrc_input_optimization_complete",
        "test_rows_used": 0,
        "folds": [int(value) for value in config.folds],
        "lead": int(config.lead),
        "fixed_ridge_alpha": float(config.ridge_alpha),
        "candidate_ranking": all_rows["architecture"].tolist(),
        "promotion_gate_passes": all_rows.loc[
            all_rows["promotion_gate"], "architecture"
        ].tolist(),
        "interpretation_guard": (
            "This is a bounded exploratory comparison. A candidate is not a QRC-specific "
            "improvement unless it also beats its matched raw-PCA control and the incumbent "
            "interacting QRC beats interaction-off and temporal-shuffle controls."
        ),
        "files": {
            "primary_comparison": "primary_comparison.csv",
            "architecture_summary": "architecture_summary.csv",
            "fold_metrics": "fold_metrics.csv",
            "validation_cells": "validation_cells.csv.gz",
            "label_gap_by_fold": "label_gap_by_fold.csv",
            "label_gap_summary": "label_gap_summary.csv",
            "readout_fits": "readout_fits.csv",
            "feature_diagnostics": "feature_diagnostics.csv",
            "input_scalers": "input_scalers.csv",
            "selected_samples": "selected_samples.csv",
        },
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir
