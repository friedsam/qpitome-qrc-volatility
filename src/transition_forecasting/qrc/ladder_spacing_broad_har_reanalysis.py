from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.ladder_spacing_assay import (
    LadderSpacingAssayConfig,
    _geometry_selection,
    probabilities_to_mode_family,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    load_rolling_fold_dataset,
)
from transition_forecasting.qrc.transition_signal_broad_har import (
    evaluate_exact_matrix_broad_har,
)
from transition_forecasting.qrc.transition_signal_readout_assay import (
    MODEL_SPECS,
    TransitionSignalAssayConfig,
    _pooled_predictions_metrics,
)


def _load_parameters(run_dir: Path) -> dict[str, object]:
    payload = json.loads((Path(run_dir) / "params.json").read_text())
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("params.json lacks parameters object")
    return parameters


def run_spacing_broad_har_reanalysis(
    *,
    spacing_run: Path,
    results_root: Path,
    run_id: str | None = None,
) -> Path:
    """Re-score existing geometry probability caches with the established all-row HAR."""
    spacing_run = Path(spacing_run)
    parameters = _load_parameters(spacing_run)
    transition_config = TransitionSignalAssayConfig(
        **dict(parameters["transition_readout"])
    )
    spacing_config = LadderSpacingAssayConfig(
        **dict(parameters["spacing_assay"])
    )
    transition_config.validate()
    spacing_config.validate()

    fold_dir = Path(str(parameters["fold_dir"]))
    dataset = load_rolling_fold_dataset(fold_dir)
    geometry_names = tuple(dict(parameters["geometry_variants"]).keys())
    readout_families = tuple(spacing_config.readout_families)
    run_dir = begin_run(
        results_root,
        {
            "spacing_source_run": str(spacing_run),
            "fold_dir": str(fold_dir),
            "transition_readout": {
                **transition_config.to_dict(),
                "har_scope": "all_selected_rows",
            },
            "spacing_assay": spacing_config.to_dict(),
            "geometry_variants": parameters["geometry_variants"],
            "coupling_normalization": parameters["coupling_normalization"],
            "har_scope": "all_selected_rows",
            "hard_negative_role": "included_in_har; excluded_from_qrc_fit_and_calibration; validation_diagnostic",
            "quantum_evolution_reused": True,
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    fold_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold in spacing_config.folds:
        selected_frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=spacing_config.leads,
            max_per_class=spacing_config.max_per_class,
            seed=spacing_config.seed,
        )
        if selected_frame["fold_split"].eq("test").any():
            raise RuntimeError("spacing reanalysis must not receive test rows")
        for geometry_name in geometry_names:
            cache_path = (
                spacing_run
                / "probability_cache"
                / geometry_name
                / f"fold_{int(fold)}.npz"
            )
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            with np.load(cache_path, allow_pickle=False) as bundle:
                probabilities = np.asarray(bundle["probabilities"], dtype=float)
                cached_ids = np.asarray(bundle["sample_id"]).astype(str)
            selected_ids = selected_frame["sample_id"].astype(str).to_numpy()
            if set(cached_ids) != set(selected_ids):
                raise RuntimeError(
                    f"fold {fold} geometry {geometry_name}: cache and panel differ"
                )
            by_id = selected_frame.set_index(
                selected_frame["sample_id"].astype(str), drop=False
            )
            frame = by_id.loc[cached_ids].reset_index(drop=True)
            for family in readout_families:
                matrix = probabilities_to_mode_family(probabilities, family)
                for base_model_name, readout_kind in MODEL_SPECS:
                    model_name = (
                        f"{geometry_name}|{family}|{base_model_name}"
                    )
                    metric_row, candidates, predictions = (
                        evaluate_exact_matrix_broad_har(
                            matrix,
                            frame=frame,
                            config=transition_config,
                            model_name=model_name,
                            readout_kind=readout_kind,
                        )
                    )
                    metric_row["geometry"] = geometry_name
                    metric_row["readout_family"] = family
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
    selection = _geometry_selection(
        pooled, transition_config, spacing_config
    )

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    inner_candidates.to_csv(run_dir / "inner_candidates.csv.gz", index=False)
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False)
    pooled.to_csv(run_dir / "pooled_metrics_by_stratum_lead.csv", index=False)
    (run_dir / "geometry_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    summary = {
        "spacing_source_run": str(spacing_run),
        "folds": list(spacing_config.folds),
        "geometries": list(geometry_names),
        "readout_families": list(readout_families),
        "har_scope": "all_selected_rows",
        "hard_negatives_used_for_har": True,
        "hard_negatives_used_for_qrc_fit_or_calibration": False,
        "quantum_evolution_reused": True,
        "test_evaluated": False,
        "selection": selection,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
