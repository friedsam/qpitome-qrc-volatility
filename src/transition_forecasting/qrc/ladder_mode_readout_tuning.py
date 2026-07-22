from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.ladder_mode_readout_tools import (
    LadderModeReadoutConfig,
    ArchitectureArchive,
    evaluate_case_family_fold,
    grouped_prediction_metrics,
    lead_label_metrics,
    load_architecture_archive,
    prediction_frame,
    select_candidate,
    validate_archive_alignment,
)


def _case_role(case: str, config: LadderModeReadoutConfig) -> str:
    if case in config.ordered_ladder_cases:
        return "ordered_ladder_candidate"
    if case in config.reference_cases:
        return "reference"
    if case in config.control_cases:
        return "ladder_control"
    return "other"


def _candidate_families(
    archive: ArchitectureArchive,
    config: LadderModeReadoutConfig,
) -> tuple[str, ...]:
    architecture = str(archive.architecture[0])
    if architecture == "ladder":
        return tuple(config.readout_families)
    return ("occupation_pca4",)


def _har_prediction_frame(
    archive: ArchitectureArchive,
    *,
    fold: int,
) -> pd.DataFrame:
    prediction = archive.har_prediction_path
    return prediction_frame(
        archive,
        prediction,
        fold=fold,
        model_name="har",
        readout_family="baseline",
        selected_lambda=0.0,
    )


def _selected_prediction(
    row: pd.Series,
    predictions_by_key: dict[tuple[int, str, str], np.ndarray],
    archives: dict[str, ArchitectureArchive],
    *,
    policy: str,
) -> pd.DataFrame:
    fold = int(row["fold"])
    case = str(row["case"])
    family = str(row["readout_family"])
    prediction = predictions_by_key[(fold, case, family)]
    return prediction_frame(
        archives[case],
        prediction,
        fold=fold,
        model_name=f"selected_ladder_{policy}",
        readout_family=family,
        selected_lambda=float(row["selected_lambda"]),
    )


def _fold_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = (
        "fold",
        "model_name",
        "case",
        "architecture",
        "control",
        "interaction_scale",
        "readout_family",
        "selected_lambda",
    )
    from transition_forecasting.qrc.representation_screen_analysis import (
        _metric_payload,
    )

    for keys, local in predictions.groupby(
        list(group_columns),
        dropna=False,
    ):
        payload = dict(zip(group_columns, keys))
        metric = _metric_payload(
            local["y_true"].to_numpy()[:, None],
            local["y_pred"].to_numpy()[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                **payload,
                **{key: float(value) for key, value in metric.items()},
                "rows": int(len(local)),
            }
        )
    return pd.DataFrame(rows)


def _pooled_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    from transition_forecasting.qrc.representation_screen_analysis import (
        _metric_payload,
    )

    rows = []
    for model_name, local in predictions.groupby("model_name"):
        metric = _metric_payload(
            local["y_true"].to_numpy()[:, None],
            local["y_pred"].to_numpy()[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                "model_name": str(model_name),
                **{key: float(value) for key, value in metric.items()},
                "rows": int(len(local)),
                "folds": int(local["fold"].nunique()),
                "cases": "|".join(sorted(local["case"].astype(str).unique())),
                "readout_families": "|".join(
                    sorted(local["readout_family"].astype(str).unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def _model_lead_label_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    from transition_forecasting.qrc.representation_screen_analysis import (
        _metric_payload,
    )

    rows = []
    for keys, local in predictions.groupby(
        ["model_name", "lead", "label"]
    ):
        model_name, lead, label = keys
        metric = _metric_payload(
            local["y_true"].to_numpy()[:, None],
            local["y_pred"].to_numpy()[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                "model_name": str(model_name),
                "lead": int(lead),
                "label": int(label),
                **{key: float(value) for key, value in metric.items()},
                "rows": int(len(local)),
                "folds": int(local["fold"].nunique()),
                "cases": "|".join(sorted(local["case"].astype(str).unique())),
                "readout_families": "|".join(
                    sorted(local["readout_family"].astype(str).unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_plots(
    run_dir: Path,
    pooled: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    l5: pd.DataFrame,
) -> None:
    primary_names = [
        "har",
        "reference_chain_interaction_off",
        "reference_chain_interacting_1p00",
        "selected_ladder_frozen",
        "selected_ladder_modes",
        "selected_ladder_full",
    ]
    primary = pooled.loc[
        pooled["model_name"].isin(primary_names)
    ].sort_values("qlike")
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.bar(primary["model_name"], primary["qlike"])
        ax.set_ylabel("Pooled outer-validation QLIKE")
        ax.set_xlabel("Nested architecture/readout model")
        ax.set_title("Ladder mode readout tuning")
        ax.tick_params(axis="x", rotation=50)
        fig.tight_layout()
        fig.savefig(run_dir / "selected_model_qlike.png", dpi=180)
        plt.close(fig)

    fold_primary = fold_metrics.loc[
        fold_metrics["model_name"].isin(primary_names)
    ].copy()
    if not fold_primary.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for model_name, local in fold_primary.groupby("model_name"):
            ordered = local.sort_values("fold")
            ax.plot(
                ordered["fold"],
                ordered["qlike"],
                marker="o",
                label=model_name,
            )
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("QLIKE")
        ax.set_xticks(sorted(fold_primary["fold"].unique()))
        ax.set_title("Nested ladder selection by outer fold")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(run_dir / "selected_model_fold_qlike.png", dpi=180)
        plt.close(fig)

    l5_primary = l5.loc[
        l5["model_name"].isin(primary_names)
        & l5["lead"].eq(5)
        & l5["label"].eq(1)
    ].sort_values("qlike")
    if not l5_primary.empty:
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.bar(l5_primary["model_name"], l5_primary["qlike"])
        ax.set_ylabel("L5 transition QLIKE")
        ax.set_xlabel("Nested architecture/readout model")
        ax.set_title("L5 transition performance after ladder readout tuning")
        ax.tick_params(axis="x", rotation=50)
        fig.tight_layout()
        fig.savefig(run_dir / "selected_model_l5_qlike.png", dpi=180)
        plt.close(fig)


def run_ladder_mode_readout_tuning(
    *,
    ladder_run_dir: Path,
    results_root: Path,
    config: LadderModeReadoutConfig,
    run_id: str | None = None,
) -> Path:
    config.validate()
    ladder_run_dir = Path(ladder_run_dir)
    archive_root = ladder_run_dir / "feature_archives"
    if not archive_root.is_dir():
        raise FileNotFoundError(archive_root)

    requested_cases = tuple(
        dict.fromkeys(
            config.reference_cases
            + config.ordered_ladder_cases
            + config.control_cases
        )
    )
    archives = {
        case: load_architecture_archive(archive_root / f"{case}.npz")
        for case in requested_cases
    }
    validate_archive_alignment(archives)
    run_dir = begin_run(
        results_root,
        {
            "ladder_run_dir": str(ladder_run_dir),
            "archive_root": str(archive_root),
            "config": config.to_dict(),
            "requested_cases": list(requested_cases),
        },
        run_id=run_id,
    )

    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    predictions_by_key: dict[tuple[int, str, str], np.ndarray] = {}
    reference_archive = archives[config.reference_cases[0]]

    for fold in config.folds:
        prediction_frames.append(
            _har_prediction_frame(reference_archive, fold=int(fold))
        )
        fold_ordered_rows: list[dict[str, object]] = []

        for case, archive in archives.items():
            for family in _candidate_families(archive, config):
                row, prediction = evaluate_case_family_fold(
                    archive,
                    fold=int(fold),
                    family=family,
                    config=config,
                )
                row["case_role"] = _case_role(case, config)
                candidate_rows.append(row)
                predictions_by_key[(int(fold), case, family)] = prediction

                model_name = (
                    f"reference_{case}"
                    if case in config.reference_cases
                    else f"candidate_{case}_{family}"
                )
                prediction_frames.append(
                    prediction_frame(
                        archive,
                        prediction,
                        fold=int(fold),
                        model_name=model_name,
                        readout_family=family,
                        selected_lambda=float(row["selected_lambda"]),
                    )
                )
                if case in config.ordered_ladder_cases:
                    fold_ordered_rows.append(row)

        ordered_frame = pd.DataFrame(fold_ordered_rows)
        for policy in ("frozen", "modes", "full"):
            chosen = select_candidate(
                ordered_frame,
                policy=policy,
            )
            selected = {
                "selection_policy": policy,
                **chosen.to_dict(),
            }
            selected_rows.append(selected)
            prediction_frames.append(
                _selected_prediction(
                    chosen,
                    predictions_by_key,
                    archives,
                    policy=policy,
                )
            )

    candidates = pd.DataFrame(candidate_rows)
    selected = pd.DataFrame(selected_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = _fold_model_metrics(predictions)
    pooled = _pooled_model_metrics(predictions)
    pooled_by_configuration = grouped_prediction_metrics(predictions)
    group_metrics = _model_lead_label_metrics(predictions)
    group_by_configuration = lead_label_metrics(predictions)

    candidates.to_csv(run_dir / "candidate_fold_metrics.csv", index=False)
    selected.to_csv(
        run_dir / "selected_ladder_configurations.csv",
        index=False,
    )
    fold_metrics.to_csv(run_dir / "model_fold_metrics.csv", index=False)
    pooled.to_csv(run_dir / "pooled_metrics.csv", index=False)
    pooled_by_configuration.to_csv(
        run_dir / "pooled_configuration_metrics.csv",
        index=False,
    )
    group_metrics.to_csv(run_dir / "lead_label_metrics.csv", index=False)
    group_by_configuration.to_csv(
        run_dir / "lead_label_configuration_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    mode_definitions = pd.DataFrame(
        [
            {
                "mode": "symmetric_constant",
                "row_parity": "symmetric",
                "longitudinal_shape": "constant",
                "interpretation": "global ladder density",
            },
            {
                "mode": "symmetric_gradient",
                "row_parity": "symmetric",
                "longitudinal_shape": "gradient",
                "interpretation": "shared left-to-right gradient",
            },
            {
                "mode": "symmetric_curvature",
                "row_parity": "symmetric",
                "longitudinal_shape": "curvature",
                "interpretation": "shared center-versus-edge curvature",
            },
            {
                "mode": "antisymmetric_constant",
                "row_parity": "antisymmetric",
                "longitudinal_shape": "constant",
                "interpretation": "top-versus-bottom row imbalance",
            },
            {
                "mode": "antisymmetric_gradient",
                "row_parity": "antisymmetric",
                "longitudinal_shape": "gradient",
                "interpretation": "difference between row gradients",
            },
            {
                "mode": "antisymmetric_curvature",
                "row_parity": "antisymmetric",
                "longitudinal_shape": "curvature",
                "interpretation": "difference between row curvatures",
            },
        ]
    )
    mode_definitions.to_csv(run_dir / "mode_definitions.csv", index=False)

    _write_plots(run_dir, pooled, fold_metrics, group_metrics)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "development_ladder_mode_readout_tuning",
                "test_rows_used": 0,
                "selection_policies": {
                    "frozen": (
                        "select interaction scale only; use occupation PCA4"
                    ),
                    "modes": (
                        "select interaction scale and explicit ladder-mode family"
                    ),
                    "full": (
                        "select interaction scale and any frozen or mode family"
                    ),
                },
                "fixed_parameters": {
                    "ridge_alpha": config.ridge_alpha,
                    "pca_components": config.pca_components,
                    "calibration": (
                        "one global lambda selected on a chronological inner holdout"
                    ),
                    "lambda_grid": list(config.global_lambdas),
                },
                "known_limitations": [
                    (
                        "Development folds only; untouched test rows are not used."
                    ),
                    (
                        "The ladder geometry, interaction-scale set, and explicit "
                        "mode families were motivated by earlier development results."
                    ),
                    (
                        "The outer folds therefore remain development estimates, "
                        "not confirmatory test performance."
                    ),
                    (
                        "Each candidate receives its own inner-selected correction "
                        "amplitude; ridge alpha remains fixed."
                    ),
                    (
                        "Explicit ladder modes are linear combinations of occupation "
                        "observables; gains indicate readout alignment, not additional "
                        "reservoir information."
                    ),
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir
