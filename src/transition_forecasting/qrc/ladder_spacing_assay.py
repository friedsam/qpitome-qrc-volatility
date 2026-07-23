from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    evolve_ladder_probe_probabilities,
    probabilities_to_occupations,
)
from transition_forecasting.qrc.ladder_mode_readout_tools import ladder_mode_weights
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    build_candidate_sequences,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)
from transition_forecasting.qrc.transition_signal_readout_assay import (
    MODEL_SPECS,
    TransitionSignalAssayConfig,
    _freeze_recommendation,
    _pooled_predictions_metrics,
    evaluate_exact_matrix,
)

ReadoutFamily = Literal["symmetric_modes", "compact_modes"]
READOUT_MODE_INDICES: dict[ReadoutFamily, tuple[int, ...]] = {
    "symmetric_modes": (0, 1, 2),
    "compact_modes": (0, 1, 3, 4),
}


@dataclass(frozen=True)
class LadderSpacingAssayConfig:
    """Predeclared compact distance assay for the frozen six-atom ladder."""

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    readout_families: tuple[ReadoutFamily, ...] = (
        "symmetric_modes",
        "compact_modes",
    )
    incumbent_interaction_scale: float = 1.25
    stronger_bottom_spacing_scale: float = 1.15
    stronger_cross_row_spacing_um: float = 8.0
    weaker_cross_row_spacing_um: float = 10.0
    challenger_calm_tolerance: float = 0.01
    seed: int = 20260722

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if not self.leads or any(int(value) < 1 for value in self.leads):
            raise ValueError("leads must be positive")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if not self.readout_families:
            raise ValueError("readout_families cannot be empty")
        unknown = set(self.readout_families).difference(READOUT_MODE_INDICES)
        if unknown:
            raise ValueError(f"unsupported readout families: {sorted(unknown)}")
        if self.incumbent_interaction_scale <= 0:
            raise ValueError("incumbent_interaction_scale must be positive")
        if self.stronger_bottom_spacing_scale <= 0:
            raise ValueError("stronger_bottom_spacing_scale must be positive")
        if min(
            self.stronger_cross_row_spacing_um,
            self.weaker_cross_row_spacing_um,
        ) <= 0:
            raise ValueError("row spacings must be positive")
        if self.stronger_cross_row_spacing_um >= self.weaker_cross_row_spacing_um:
            raise ValueError("stronger cross-row mixing must use smaller spacing")
        if self.challenger_calm_tolerance < 0:
            raise ValueError("challenger_calm_tolerance must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def geometry_variants(
    incumbent: StaggeredLadderGeometryConfig,
    config: LadderSpacingAssayConfig,
) -> dict[str, StaggeredLadderGeometryConfig]:
    """Return the predeclared distance challengers; no broad geometry search."""
    variants = {
        "incumbent": incumbent,
        "no_displaced_defect": replace(incumbent, defect_dx_um=0.0, defect_dy_um=0.0),
        "stronger_bottom_spacing": replace(
            incumbent,
            bottom_spacing_scale=float(config.stronger_bottom_spacing_scale),
        ),
        "stronger_cross_row_mixing": replace(
            incumbent,
            row_spacing_um=float(config.stronger_cross_row_spacing_um),
        ),
        "weaker_cross_row_mixing": replace(
            incumbent,
            row_spacing_um=float(config.weaker_cross_row_spacing_um),
        ),
    }
    for geometry in variants.values():
        geometry.validate()
    return variants


def median_nearest_coupling(
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float,
) -> float:
    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    values = np.asarray(
        [
            precomputed.interaction_matrix[left, right]
            for left, right in precomputed.nearest_pairs
        ],
        dtype=float,
    )
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("nearest-neighbor coupling set is invalid")
    return float(np.median(values))


def normalized_interaction_scale(
    reservoir: TemporalRydbergChainConfig,
    incumbent: StaggeredLadderGeometryConfig,
    challenger: StaggeredLadderGeometryConfig,
    *,
    incumbent_scale: float,
) -> float:
    """Hold median nearest-neighbor coupling fixed across distance variants."""
    target = median_nearest_coupling(
        reservoir,
        incumbent,
        interaction_scale=float(incumbent_scale),
    )
    unscaled = median_nearest_coupling(
        reservoir,
        challenger,
        interaction_scale=1.0,
    )
    return float(target / unscaled)


def probabilities_to_mode_family(
    probabilities: np.ndarray,
    family: ReadoutFamily,
) -> np.ndarray:
    if family not in READOUT_MODE_INDICES:
        raise ValueError(f"unsupported readout family: {family}")
    occupations = probabilities_to_occupations(probabilities)
    weights = ladder_mode_weights()
    all_modes = np.einsum("rpa,am->rpm", occupations, weights)
    selected = READOUT_MODE_INDICES[family]
    return all_modes[:, :, selected].reshape(len(occupations), -1)


def _load_source_parameters(source_run: Path) -> dict[str, object]:
    payload = json.loads((Path(source_run) / "params.json").read_text())
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("source params.json lacks parameters object")
    return parameters


def _geometry_selection(
    pooled: pd.DataFrame,
    transition_config: TransitionSignalAssayConfig,
    spacing_config: LadderSpacingAssayConfig,
) -> dict[str, object]:
    recommendation = _freeze_recommendation(pooled, transition_config)
    candidates = recommendation["candidates"]
    incumbent = [
        row
        for row in candidates
        if str(row["model_name"]).startswith("incumbent|")
    ]
    if not incumbent:
        raise RuntimeError("distance assay lacks incumbent candidates")
    incumbent_best = min(
        incumbent,
        key=lambda row: (
            0 if bool(row["eligible"]) else 1,
            -float(row["transition_l5_qlike_gain"]),
            float(row["calm_qlike_delta"]),
        ),
    )
    eligible_challengers: list[dict[str, object]] = []
    for row in candidates:
        if str(row["model_name"]).startswith("incumbent|"):
            continue
        if not bool(row["eligible"]):
            continue
        if float(row["calm_qlike_delta"]) > (
            float(incumbent_best["calm_qlike_delta"])
            + spacing_config.challenger_calm_tolerance
        ):
            continue
        if float(row["transition_l5_qlike_gain"]) <= float(
            incumbent_best["transition_l5_qlike_gain"]
        ):
            continue
        eligible_challengers.append(row)
    selected = (
        max(
            eligible_challengers,
            key=lambda row: (
                float(row["transition_l5_qlike_gain"]),
                float(row["transition_calm_correction_margin"]),
                -float(row["calm_qlike_delta"]),
            ),
        )
        if eligible_challengers
        else incumbent_best
    )
    selected_is_challenger = selected is not incumbent_best
    return {
        "incumbent": incumbent_best,
        "selected": selected,
        "selected_is_challenger": bool(selected_is_challenger),
        "challenger_requires_1000_shot_confirmation": bool(selected_is_challenger),
        "rule": (
            "replace incumbent only for larger transition-L5 QLIKE gain with no "
            "more than the predeclared calm deterioration allowance; exact winner "
            "requires 1000-shot confirmation"
        ),
        "all_model_candidates": candidates,
    }


def run_ladder_spacing_assay(
    *,
    source_run: Path,
    results_root: Path,
    transition_config: TransitionSignalAssayConfig = TransitionSignalAssayConfig(),
    spacing_config: LadderSpacingAssayConfig = LadderSpacingAssayConfig(),
    run_id: str | None = None,
) -> Path:
    transition_config.validate()
    spacing_config.validate()
    if tuple(transition_config.folds) != tuple(spacing_config.folds):
        raise ValueError("transition and spacing fold contracts differ")
    source_run = Path(source_run)
    parameters = _load_source_parameters(source_run)
    fold_dir = Path(str(parameters["fold_dir"]))
    dataset = load_rolling_fold_dataset(fold_dir)
    reservoir = TemporalRydbergChainConfig(**dict(parameters["reservoir"]))
    reservoir.validate()
    if reservoir.shots is not None:
        reservoir = replace(reservoir, shots=None)
    candidate_features = CandidateFeatureConfig(**dict(parameters["candidate_features"]))
    candidate_features.validate()
    incumbent = StaggeredLadderGeometryConfig(**dict(parameters["ladder_geometry"]))
    incumbent.validate()
    variants = geometry_variants(incumbent, spacing_config)
    level_channel = resolve_level_channel(
        dataset,
        name=str(parameters["study"]["level_channel_name"]),
        fallback=int(parameters["study"]["fallback_level_channel"]),
    )
    run_dir = begin_run(
        results_root,
        {
            "source_run": str(source_run),
            "fold_dir": str(fold_dir),
            "transition_readout": transition_config.to_dict(),
            "spacing_assay": spacing_config.to_dict(),
            "geometry_variants": {
                name: geometry.to_dict() for name, geometry in variants.items()
            },
            "coupling_normalization": "fixed_median_nearest_neighbor_coupling",
            "hard_negative_role": "validation_diagnostic_only",
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    fold_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    geometry_rows: list[dict[str, object]] = []
    target_coupling = median_nearest_coupling(
        reservoir,
        incumbent,
        interaction_scale=spacing_config.incumbent_interaction_scale,
    )

    for fold in spacing_config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=spacing_config.leads,
            max_per_class=spacing_config.max_per_class,
            seed=spacing_config.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("distance assay must not receive test rows")
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        source = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=int(parameters["study"]["sequence_length"]),
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(source).all(axis=1)
        frame = frame.loc[usable].reset_index(drop=True)
        source = source[usable]
        train = frame["fold_split"].eq("train").to_numpy()
        raw_sequence = build_candidate_sequences(
            source,
            "level_instability",
            candidate_features,
        )
        scaler = fit_channel_scaler(raw_sequence, train)
        encoded = transform_candidate_sequences(raw_sequence, scaler)

        for geometry_name, geometry in variants.items():
            interaction_scale = normalized_interaction_scale(
                reservoir,
                incumbent,
                geometry,
                incumbent_scale=spacing_config.incumbent_interaction_scale,
            )
            probabilities, metadata = evolve_ladder_probe_probabilities(
                encoded,
                reservoir,
                geometry,
                interaction_scale=interaction_scale,
                condition="ordered",
            )
            cache_path = (
                run_dir
                / "probability_cache"
                / geometry_name
                / f"fold_{int(fold)}.npz"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                probabilities=probabilities,
                sample_id=frame["sample_id"].astype(str).to_numpy(),
                fold_split=frame["fold_split"].astype(str).to_numpy(),
                lead=frame["lead"].to_numpy(dtype=int),
                evaluation_stratum=frame["evaluation_stratum"].astype(str).to_numpy(),
            )
            if int(fold) == int(spacing_config.folds[0]):
                geometry_rows.append(
                    {
                        "geometry": geometry_name,
                        "interaction_scale": float(interaction_scale),
                        "target_median_nearest_coupling": float(target_coupling),
                        "actual_median_nearest_coupling": median_nearest_coupling(
                            reservoir,
                            geometry,
                            interaction_scale=interaction_scale,
                        ),
                        "positions_um": json.dumps(metadata["positions_um"]),
                        "geometry_config": json.dumps(geometry.to_dict(), sort_keys=True),
                    }
                )
            for family in spacing_config.readout_families:
                matrix = probabilities_to_mode_family(probabilities, family)
                for base_model_name, readout_kind in MODEL_SPECS:
                    model_name = f"{geometry_name}|{family}|{base_model_name}"
                    metric_row, candidates, predictions = evaluate_exact_matrix(
                        matrix,
                        frame=frame,
                        config=transition_config,
                        model_name=model_name,
                        readout_kind=readout_kind,
                    )
                    metric_row["geometry"] = geometry_name
                    metric_row["readout_family"] = family
                    metric_row["interaction_scale"] = float(interaction_scale)
                    fold_rows.append(metric_row)
                    candidates.insert(2, "geometry", geometry_name)
                    candidates.insert(3, "readout_family", family)
                    candidate_frames.append(candidates)
                    predictions.insert(1, "geometry", geometry_name)
                    predictions.insert(2, "readout_family", family)
                    prediction_frames.append(predictions)

    fold_metrics = pd.DataFrame(fold_rows)
    inner_candidates = pd.concat(candidate_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = _pooled_predictions_metrics(predictions)
    selection = _geometry_selection(pooled, transition_config, spacing_config)
    geometry_summary = pd.DataFrame(geometry_rows)

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    inner_candidates.to_csv(run_dir / "inner_candidates.csv.gz", index=False)
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False)
    pooled.to_csv(run_dir / "pooled_metrics_by_stratum_lead.csv", index=False)
    geometry_summary.to_csv(run_dir / "geometry_summary.csv", index=False)
    (run_dir / "geometry_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    summary = {
        "source_run": str(source_run),
        "folds": list(spacing_config.folds),
        "geometries": list(variants),
        "readout_families": list(spacing_config.readout_families),
        "target_median_nearest_coupling": float(target_coupling),
        "hard_negatives_used_for_fit_or_calibration": False,
        "test_evaluated": False,
        "selection": selection,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
