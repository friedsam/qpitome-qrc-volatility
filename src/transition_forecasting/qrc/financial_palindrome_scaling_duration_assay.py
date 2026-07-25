from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.financial_qrc_feature_transfer_assay import (
    _cell_frame,
    _directional_payload,
    _fit_signed_residual_probe,
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
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    _select_rows_for_fold,
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


@dataclass(frozen=True)
class FinancialPalindromeScalingDurationConfig:
    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    lead: int = 5
    max_per_class: int = 12
    sequence_length: int = 40
    prequential_blocks: int = 5
    ridge_alpha: float = 100.0
    min_causal_residual_rows: int = 10
    selection_seed: int = 20260721
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    split_horizon: int = 4
    durations_us: tuple[float, ...] = (0.015, 0.020, 0.025)
    scaling_quantiles: tuple[tuple[float, float], ...] = (
        (0.005, 0.995),
        (0.010, 0.990),
        (0.025, 0.975),
        (0.050, 0.950),
    )
    interaction_scale: float = 1.25
    palindrome_schedule: str = "crossover_Ahalf_B_Ahalf"

    def validate(self) -> None:
        if self.lead != 5:
            raise ValueError("assay is intentionally restricted to L5")
        if not self.folds or any(fold <= 3 for fold in self.folds):
            raise ValueError("folds must be development folds 4+")
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be unique")
        if self.max_per_class < 1 or self.sequence_length < 2:
            raise ValueError("invalid sample cap or sequence length")
        if self.prequential_blocks < 2 or self.min_causal_residual_rows < 5:
            raise ValueError("invalid causal residual configuration")
        if self.ridge_alpha <= 0 or self.interaction_scale <= 0:
            raise ValueError("ridge alpha and interaction scale must be positive")
        if not self.durations_us or any(value <= 0 for value in self.durations_us):
            raise ValueError("durations must be positive")
        if len(set(self.durations_us)) != len(self.durations_us):
            raise ValueError("durations must be unique")
        if not self.scaling_quantiles:
            raise ValueError("scaling quantiles must be nonempty")
        for low, high in self.scaling_quantiles:
            if not 0.0 <= low < high <= 1.0:
                raise ValueError("invalid scaling quantiles")
        names = {schedule.name for schedule in CROSSOVER_SCHEDULES}
        if self.palindrome_schedule not in names:
            raise ValueError("unknown palindrome schedule")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _schedule(name: str):
    for schedule in CROSSOVER_SCHEDULES:
        if schedule.name == name:
            return schedule
    raise ValueError(f"unknown crossover schedule: {name}")


def _candidate_name(q_low: float, q_high: float, duration_us: float, interaction: str) -> str:
    return (
        f"q{q_low:.3f}_{q_high:.3f}__dt{duration_us:.3f}__{interaction}"
        .replace(".", "p")
    )


def _encoding_diagnostics(encoded: np.ndarray, train: np.ndarray, validation: np.ndarray) -> dict[str, float]:
    values = np.asarray(encoded, dtype=float)
    payload: dict[str, float] = {}
    for channel in (0, 1):
        for split_name, mask in (("train", train), ("validation", validation)):
            local = values[mask, :, channel]
            payload[f"channel{channel}_{split_name}_mean"] = float(np.mean(local))
            payload[f"channel{channel}_{split_name}_std"] = float(np.std(local))
            payload[f"channel{channel}_{split_name}_clip_fraction"] = float(
                np.mean(np.abs(local) >= 1.0 - 1e-12)
            )
            payload[f"channel{channel}_{split_name}_q01"] = float(np.quantile(local, 0.01))
            payload[f"channel{channel}_{split_name}_q99"] = float(np.quantile(local, 0.99))
    return payload


def _aggregate(candidate_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["q_low", "q_high", "duration_us", "interaction"]
    for key, group in candidate_metrics.groupby(keys, sort=True):
        q_low, q_high, duration_us, interaction = key
        rows.append(
            {
                "q_low": float(q_low),
                "q_high": float(q_high),
                "duration_us": float(duration_us),
                "interaction": str(interaction),
                "folds": int(group["fold"].nunique()),
                "qlike_fold_wins": int(group["qlike_delta"].lt(0.0).sum()),
                "rmse_fold_wins": int(group["rmse_delta"].lt(0.0).sum()),
                "positive_correlation_folds": int(
                    group["correction_residual_correlation"].gt(0.0).sum()
                ),
                "positive_sign_gap_folds": int(group["residual_sign_gap"].gt(0.0).sum()),
                "mean_qlike_delta": float(group["qlike_delta"].mean()),
                "mean_rmse_delta": float(group["rmse_delta"].mean()),
                "mean_correlation": float(group["correction_residual_correlation"].mean()),
                "mean_balanced_sign_accuracy": float(group["balanced_sign_accuracy"].mean()),
                "mean_residual_sign_gap": float(group["residual_sign_gap"].mean()),
                "mean_wrong_up_rate": float(group["wrong_up_rate"].mean()),
                "mean_wrong_down_rate": float(group["wrong_down_rate"].mean()),
                "mean_abs_correction": float(group["mean_abs_correction"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        [
            "positive_correlation_folds",
            "positive_sign_gap_folds",
            "qlike_fold_wins",
            "rmse_fold_wins",
            "mean_qlike_delta",
            "mean_rmse_delta",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def run_financial_palindrome_scaling_duration_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    config: FinancialPalindromeScalingDurationConfig = FinancialPalindromeScalingDurationConfig(),
    candidate_features: CandidateFeatureConfig = CandidateFeatureConfig(),
    reservoir: TemporalRydbergChainConfig | None = None,
    geometry: StaggeredLadderGeometryConfig | None = None,
    run_id: str | None = None,
) -> Path:
    config.validate()
    candidate_features.validate()
    dataset = load_rolling_fold_dataset(Path(fold_dir))
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
        step_duration_us=0.020,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=config.selection_seed,
    )
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    base_reservoir.validate()
    geometry_config.validate()
    if base_reservoir.n_atoms != 6 or base_reservoir.shots is not None:
        raise ValueError("assay requires exact six-atom simulation")

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "config": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_test_rows_used": 0,
            "selection_scope": "development validation folds only",
            "fit_intercept": False,
            "ridge_alpha": config.ridge_alpha,
            "scientific_question": (
                "Which leakage-safe input scaling and local palindrome duration transfer best "
                "from synthetic capacity to signed L5 financial residual correction?"
            ),
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=False)

    metric_rows: list[dict[str, object]] = []
    scaling_rows: list[dict[str, object]] = []
    retained_rows: list[pd.DataFrame] = []

    for fold in config.folds:
        selected = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=(config.lead,),
            max_per_class=config.max_per_class,
            excluded_ids=set(),
            seed=config.selection_seed,
        )
        tensor_rows = selected["_tensor_row"].to_numpy(dtype=int)
        level = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=config.sequence_length,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(level).all(axis=1)
        frame = selected.loc[usable].reset_index(drop=True)
        level = level[usable]
        if frame.empty or frame["fold_split"].eq("test").any():
            raise RuntimeError(f"fold {fold}: invalid development selection")
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        if not train.any() or not validation.any():
            raise RuntimeError(f"fold {fold}: empty train or validation")

        raw_sequences = build_candidate_sequences(level, "level_instability", candidate_features)
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train)
        residuals, residual_train_mask = _prequential_har_residuals(
            frame, y, train, blocks=config.prequential_blocks
        )
        if int(residual_train_mask.sum()) < config.min_causal_residual_rows:
            raise RuntimeError(f"fold {fold}: insufficient causal residual rows")

        retained = frame[["sample_id", "episode_id", "origin_date", "label", "lead", "fold_split"]].copy()
        retained.insert(0, "fold", int(fold))
        retained_rows.append(retained)

        for q_low, q_high in config.scaling_quantiles:
            scaler = fit_channel_scaler(
                raw_sequences,
                train,
                q_low=float(q_low),
                q_high=float(q_high),
            )
            encoded = transform_candidate_sequences(raw_sequences, scaler)
            scaling_rows.append(
                {
                    "fold": int(fold),
                    "q_low": float(q_low),
                    "q_high": float(q_high),
                    "channel0_median": float(scaler.medians[0]),
                    "channel1_median": float(scaler.medians[1]),
                    "channel0_half_range": float(scaler.half_ranges[0]),
                    "channel1_half_range": float(scaler.half_ranges[1]),
                    **_encoding_diagnostics(encoded, train, validation),
                }
            )

            for duration_us in config.durations_us:
                for interaction in ("off", "on"):
                    scale = config.interaction_scale if interaction == "on" else 0.0
                    reservoir_config = replace(
                        base_reservoir,
                        step_duration_us=float(duration_us),
                        shot_seed=config.selection_seed + int(fold),
                    )
                    probabilities, _ = evolve_crossover_probabilities(
                        encoded,
                        reservoir_config,
                        geometry_config,
                        _schedule(config.palindrome_schedule),
                        interaction_scale=scale,
                        drive_phase_rad=0.0,
                    )
                    features = build_crossover_feature_banks(probabilities)["occupation_pair_raw"]
                    correction, diagnostics, _ = _fit_signed_residual_probe(
                        features,
                        y=y,
                        har=har,
                        residuals=residuals,
                        residual_train_mask=residual_train_mask,
                        origin_date=frame["origin_date"].astype(str).to_numpy(),
                        config=type("ProbeConfig", (), {
                            "ridge_alpha": config.ridge_alpha,
                            "min_causal_residual_rows": config.min_causal_residual_rows,
                        })(),
                        pc1_only=False,
                    )
                    architecture = _candidate_name(q_low, q_high, duration_us, interaction)
                    cells = _cell_frame(
                        frame,
                        y,
                        har,
                        correction,
                        validation,
                        fold=int(fold),
                        architecture=architecture,
                        split_horizon=config.split_horizon,
                    )
                    payload = _directional_payload(cells)
                    metric_rows.append(
                        {
                            "fold": int(fold),
                            "q_low": float(q_low),
                            "q_high": float(q_high),
                            "duration_us": float(duration_us),
                            "interaction": interaction,
                            "architecture": architecture,
                            "feature_width": int(features.shape[1]),
                            "coefficient_l2": float(diagnostics["coefficient_l2"]),
                            **payload,
                        }
                    )
                    np.savez_compressed(
                        probability_dir / f"fold_{fold}__{architecture}.npz",
                        probabilities=probabilities,
                        encoded_sequences=encoded,
                        sample_id=frame["sample_id"].astype(str).to_numpy(),
                        fold_split=frame["fold_split"].astype(str).to_numpy(),
                    )

    candidate_metrics = pd.DataFrame(metric_rows)
    aggregate = _aggregate(candidate_metrics)
    scaling = pd.DataFrame(scaling_rows)
    retained = pd.concat(retained_rows, ignore_index=True)

    candidate_metrics.to_csv(run_dir / "candidate_fold_metrics.csv", index=False)
    aggregate.to_csv(run_dir / "candidate_summary.csv", index=False)
    scaling.to_csv(run_dir / "encoding_scaling_diagnostics.csv", index=False)
    retained.to_csv(run_dir / "retained_rows.csv", index=False)

    best = aggregate.iloc[0].to_dict() if not aggregate.empty else {}
    frozen = aggregate.loc[
        aggregate["q_low"].eq(0.01)
        & aggregate["q_high"].eq(0.99)
        & aggregate["duration_us"].eq(0.02)
        & aggregate["interaction"].eq("on")
    ]
    summary = {
        "status": "financial_palindrome_scaling_duration_complete",
        "candidates_completed": int(len(aggregate)),
        "folds_completed": int(candidate_metrics["fold"].nunique()),
        "best_ranked_candidate": best,
        "frozen_synthetic_transfer_candidate": (
            frozen.iloc[0].to_dict() if not frozen.empty else None
        ),
        "promotion_rule": (
            "Do not promote from mean loss alone. Require positive residual correlation and "
            "positive residual-sign gap in at least three folds, with no broad QLIKE/RMSE damage."
        ),
        "next_parameter_family": (
            "probe fractions only if one scaling-duration candidate shows stable directional value; "
            "otherwise stop physics retuning and report failed transfer"
        ),
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir
