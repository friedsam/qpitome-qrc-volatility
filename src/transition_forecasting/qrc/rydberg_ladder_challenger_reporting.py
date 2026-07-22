from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transition_forecasting.qrc.representation_screen_analysis import (
    _metric_payload,
)
from transition_forecasting.qrc.rydberg_ladder_challenger_tools import (
    ArchitectureCase,
    LadderChallengerConfig,
    _write_case_archive,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    ladder_pair_groups,
    staggered_ladder_positions,
)


def _write_geometry_plot(
    geometry: StaggeredLadderGeometryConfig,
    path: Path,
) -> None:
    positions = staggered_ladder_positions(geometry)
    groups = ladder_pair_groups()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for group_name in ("row", "rung", "diagonal"):
        for left, right in groups[group_name]:
            ax.plot(
                positions[[left, right], 0],
                positions[[left, right], 1],
                linewidth=1.2,
                alpha=0.75,
            )
    ax.scatter(positions[:, 0], positions[:, 1], s=100)
    for site, (x_value, y_value) in enumerate(positions):
        ax.text(x_value, y_value + 0.55, str(site), ha="center")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Six-atom staggered asymmetric ladder")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def finalize_ladder_challenger_run(
    *,
    run_dir: Path,
    cases: tuple[ArchitectureCase, ...],
    challenger: LadderChallengerConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    fold_metric_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    diagnostic_rows: list[dict[str, object]],
    ladder_mode_rows: list[dict[str, object]],
    prediction_frames: list[pd.DataFrame],
    equivalence_rows: list[dict[str, object]],
    path_blocks: dict[tuple[str, str], list[dict[str, np.ndarray]]],
    archive_blocks: dict[str, list[dict[str, object]]],
    archive_names: dict[str, tuple[str, ...]],
    architecture_metadata: dict[str, dict[str, object]],
) -> Path:
    fold_metrics = pd.DataFrame(fold_metric_rows)
    group_metrics = pd.DataFrame(group_rows)
    feature_diagnostics = pd.DataFrame(diagnostic_rows)
    mode_diagnostics = pd.DataFrame(ladder_mode_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    equivalence = pd.DataFrame(equivalence_rows)

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    group_metrics.to_csv(run_dir / "group_metrics.csv", index=False)
    feature_diagnostics.to_csv(
        run_dir / "feature_diagnostics.csv", index=False
    )
    mode_diagnostics.to_csv(
        run_dir / "ladder_mode_diagnostics.csv", index=False
    )
    equivalence.to_csv(
        run_dir / "noninteracting_equivalence.csv", index=False
    )
    predictions.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = (
        fold_metrics.groupby(
            [
                "case",
                "architecture",
                "control",
                "interaction_scale",
                "readout_variant",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_val_qlike=("val_qlike", "mean"),
            sd_val_qlike=("val_qlike", "std"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_val_mz_r2=("val_mz_r2", "mean"),
            mean_val_mz_slope=("val_mz_slope", "mean"),
            mean_prediction_std=("val_prediction_std", "mean"),
            mean_selected_lambda=("selected_lambda", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(["readout_variant", "mean_val_qlike"])
    )
    summary.to_csv(run_dir / "case_summary.csv", index=False)

    pooled_rows: list[dict[str, object]] = []
    for (case_name, variant), blocks in path_blocks.items():
        y_values = np.concatenate([block["y"] for block in blocks])
        prediction_values = np.concatenate(
            [block["prediction"] for block in blocks]
        )
        case_match = next(
            (case for case in cases if case.name == case_name), None
        )
        pooled_rows.append(
            {
                "case": case_name,
                "architecture": (
                    "classical"
                    if case_match is None
                    else case_match.architecture
                ),
                "control": "har" if case_match is None else case_match.control,
                "interaction_scale": (
                    np.nan if case_match is None else case_match.interaction_scale
                ),
                "readout_variant": variant,
                **_metric_payload(
                    y_values,
                    prediction_values,
                    np.ones(len(y_values), dtype=bool),
                ),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values(
        ["readout_variant", "qlike"]
    )
    pooled.to_csv(run_dir / "pooled_metrics.csv", index=False)

    l5 = predictions.loc[
        predictions["lead"].eq(5)
        & predictions["label"].eq(1)
    ].copy()
    l5_rows = []
    for keys, local in l5.groupby(
        [
            "case",
            "architecture",
            "control",
            "interaction_scale",
            "readout_variant",
        ],
        dropna=False,
    ):
        payload = dict(
            zip(
                [
                    "case",
                    "architecture",
                    "control",
                    "interaction_scale",
                    "readout_variant",
                ],
                keys,
            )
        )
        payload.update(
            _metric_payload(
                local["y_true"].to_numpy()[:, None],
                local["y_pred"].to_numpy()[:, None],
                np.ones(len(local), dtype=bool),
            )
        )
        l5_rows.append(payload)
    l5_summary = pd.DataFrame(l5_rows).sort_values(
        ["readout_variant", "qlike"]
    )
    l5_summary.to_csv(run_dir / "l5_transition_summary.csv", index=False)

    archive_root = run_dir / "feature_archives"
    archive_root.mkdir()
    for case in cases:
        _write_case_archive(
            archive_root / f"{case.name}.npz",
            archive_blocks[case.name],
            archive_names[case.name],
        )
    (run_dir / "architecture_metadata.json").write_text(
        json.dumps(architecture_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    calibrated = pooled.loc[
        pooled["readout_variant"].isin(["baseline", "calibrated"])
    ].sort_values("qlike")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(calibrated["case"], calibrated["qlike"])
    ax.set_ylabel("Pooled validation QLIKE")
    ax.set_xlabel("Architecture case")
    ax.set_title("Calibrated chain versus staggered-ladder challenger")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(run_dir / "architecture_qlike.png", dpi=180)
    plt.close(fig)

    ladder_sweep = pooled.loc[
        pooled["case"].str.startswith("ladder_ordered_")
        & pooled["readout_variant"].eq("calibrated")
    ].sort_values("interaction_scale")
    if not ladder_sweep.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            ladder_sweep["interaction_scale"],
            ladder_sweep["qlike"],
            marker="o",
            label="Ladder",
        )
        chain_on = pooled.loc[
            pooled["case"].eq("chain_interacting_1p00")
            & pooled["readout_variant"].eq("calibrated")
        ]
        chain_off = pooled.loc[
            pooled["case"].eq("chain_interaction_off")
            & pooled["readout_variant"].eq("calibrated")
        ]
        if not chain_on.empty:
            ax.axhline(
                float(chain_on.iloc[0]["qlike"]),
                linestyle="--",
                linewidth=1.2,
                label="Interacting chain",
            )
        if not chain_off.empty:
            ax.axhline(
                float(chain_off.iloc[0]["qlike"]),
                linestyle=":",
                linewidth=1.2,
                label="Interactions off",
            )
        ax.set_xlabel("Ladder interaction scale")
        ax.set_ylabel("Pooled validation QLIKE")
        ax.set_title("Narrow ladder interaction sweep")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "ladder_interaction_sweep.png", dpi=180)
        plt.close(fig)

    calibrated_l5 = l5_summary.loc[
        l5_summary["readout_variant"].isin(["baseline", "calibrated"])
    ].sort_values("qlike")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(calibrated_l5["case"], calibrated_l5["qlike"])
    ax.set_ylabel("L5 transition QLIKE")
    ax.set_xlabel("Architecture case")
    ax.set_title("L5 transition comparison")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(run_dir / "l5_transition_qlike.png", dpi=180)
    plt.close(fig)

    _write_geometry_plot(
        ladder_geometry,
        run_dir / "ladder_geometry.png",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "development_ladder_architecture_challenger",
                "test_rows_used": 0,
                "frozen_readout": {
                    "observable_family": "occupation",
                    "pca_components": challenger.pca_components,
                    "ridge_alpha": challenger.ridge_alpha,
                    "calibration": "one global lambda selected on a chronological inner holdout",
                    "lambda_grid": list(challenger.global_lambdas),
                },
                "cases": [case.to_dict() for case in cases],
                "known_limitations": [
                    "Development folds only; untouched test rows are not used.",
                    (
                        "The instability input and frozen readout were selected "
                        "during earlier development assays."
                    ),
                    (
                        "The ladder geometry and narrow interaction scales are "
                        "exploratory architecture choices."
                    ),
                    "Each architecture receives its own inner-selected calibration factor.",
                    (
                        "The ladder uses global drives; geometry contributes "
                        "only through interactions."
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
