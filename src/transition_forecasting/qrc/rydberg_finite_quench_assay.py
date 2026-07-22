from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    TARGET_COLUMNS,
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
from transition_forecasting.qrc.rydberg_finite_quench_tools import (
    FiniteQuenchAssayConfig,
    evolve_finite_quench,
    finite_quench_feature_blocks,
    finite_quench_feature_families,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.rydberg_susceptibility_tools import (
    classical_context_features,
    feature_diagnostics,
    fit_warning_classifier,
    long_feature_frame,
    prepare_ladder_history,
    reporter_geometry,
    warning_metrics,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


PRIMARY_GEOMETRY = "reporter_undisplaced"
PRIMARY_FAMILY = "classical_plus_reporter_quench"


def _prediction_frame(
    frame: pd.DataFrame,
    probability: np.ndarray,
    validation_mask: np.ndarray,
    *,
    fold: int,
    geometry_case: str,
    estimator: str,
    shot_budget: int,
    replicate: int,
    feature_family: str,
) -> pd.DataFrame:
    selected = frame.loc[validation_mask].reset_index(drop=True)
    return selected[
        ["sample_id", "label", "episode_id", "origin_date"]
    ].assign(
        fold=int(fold),
        geometry_case=geometry_case,
        estimator=estimator,
        shot_budget=int(shot_budget),
        replicate=int(replicate),
        feature_family=feature_family,
        transition_probability=np.asarray(probability)[validation_mask],
    )


def _metric_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    validation_mask: np.ndarray,
    train_mask: np.ndarray,
    *,
    fold: int,
    geometry_case: str,
    estimator: str,
    shot_budget: int,
    replicate: int,
    feature_family: str,
    diagnostics: dict[str, float],
) -> dict[str, object]:
    return {
        "fold": int(fold),
        "geometry_case": geometry_case,
        "estimator": estimator,
        "shot_budget": int(shot_budget),
        "replicate": int(replicate),
        "feature_family": feature_family,
        "train_rows": int(np.asarray(train_mask, dtype=bool).sum()),
        "validation_rows": int(validation_mask.sum()),
        **diagnostics,
        **warning_metrics(
            frame["label"].to_numpy(dtype=int),
            probability,
            validation_mask,
        ),
    }


def _pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [
        "geometry_case",
        "estimator",
        "shot_budget",
        "replicate",
        "feature_family",
    ]
    for keys, local in predictions.groupby(group_columns, dropna=False):
        labels = local["label"].to_numpy(dtype=int)
        probabilities = local["transition_probability"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "folds": int(local["fold"].nunique()),
                "samples": int(len(local)),
                **warning_metrics(
                    labels,
                    probabilities,
                    np.ones(len(local), dtype=bool),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["average_precision", "roc_auc"],
        ascending=[False, False],
    )


def _shot_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "average_precision",
        "roc_auc",
        "brier",
        "log_loss",
        "precision_at_0p5",
        "recall_at_0p5",
        "false_positive_rate_at_0p5",
        "mean_probability_positive",
        "mean_probability_control",
    ]
    rows: list[dict[str, object]] = []
    groups = [
        "geometry_case",
        "estimator",
        "shot_budget",
        "feature_family",
    ]
    for keys, local in pooled.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row["replicates"] = int(len(local))
        for column in metric_columns:
            row[f"{column}_mean"] = float(local[column].mean())
            row[f"{column}_std"] = float(local[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "average_precision_mean",
        ascending=False,
    )


def _geometry_effect(pooled: pd.DataFrame) -> pd.DataFrame:
    qrc = pooled.loc[
        pooled["geometry_case"].isin(
            ["reporter_undisplaced", "reporter_displaced"]
        )
    ].copy()
    keys = ["estimator", "shot_budget", "replicate", "feature_family"]
    primary = qrc.loc[
        qrc["geometry_case"].eq("reporter_undisplaced")
    ].set_index(keys)
    control = qrc.loc[
        qrc["geometry_case"].eq("reporter_displaced")
    ].set_index(keys)
    joined = primary.join(
        control,
        how="inner",
        lsuffix="_undisplaced",
        rsuffix="_displaced",
    ).reset_index()
    if joined.empty:
        return joined
    joined["delta_average_precision_undisplaced_minus_displaced"] = (
        joined["average_precision_undisplaced"]
        - joined["average_precision_displaced"]
    )
    joined["delta_roc_auc_undisplaced_minus_displaced"] = (
        joined["roc_auc_undisplaced"] - joined["roc_auc_displaced"]
    )
    joined["delta_brier_undisplaced_minus_displaced"] = (
        joined["brier_undisplaced"] - joined["brier_displaced"]
    )
    return joined


def _feature_fidelity(feature_values: pd.DataFrame) -> pd.DataFrame:
    exact = feature_values.loc[
        feature_values["estimator"].eq("expectation")
        & feature_values["shot_budget"].eq(0)
        & feature_values["replicate"].eq(0)
    ].copy()
    sampled = feature_values.loc[
        feature_values["estimator"].eq("shots")
    ].copy()
    keys = [
        "fold",
        "geometry_case",
        "sample_id",
        "label",
        "episode_id",
        "origin_date",
        "fold_split",
        "feature_family",
        "feature_name",
    ]
    joined = sampled.merge(
        exact[keys + ["feature_value"]],
        on=keys,
        how="inner",
        suffixes=("_sampled", "_exact"),
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    groups = [
        "geometry_case",
        "shot_budget",
        "replicate",
        "feature_family",
    ]
    for group_keys, local in joined.groupby(groups, dropna=False):
        observed = local["feature_value_exact"].to_numpy(dtype=float)
        estimated = local["feature_value_sampled"].to_numpy(dtype=float)
        observed_std = float(np.std(observed))
        correlation = (
            float(np.corrcoef(observed, estimated)[0, 1])
            if observed_std > 1e-12 and np.std(estimated) > 1e-12
            else 0.0
        )
        rmse = float(np.sqrt(np.mean((estimated - observed) ** 2)))
        rows.append(
            {
                **dict(zip(groups, group_keys)),
                "values": int(len(local)),
                "correlation_with_expectation": correlation,
                "feature_rmse": rmse,
                "expectation_std": observed_std,
                "noise_to_signal": (
                    rmse / observed_std if observed_std > 1e-12 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["geometry_case", "shot_budget", "feature_family", "replicate"]
    )


def _lookup_ap(
    pooled: pd.DataFrame,
    *,
    geometry_case: str,
    estimator: str,
    shot_budget: int,
    replicate: int,
    feature_family: str,
) -> float:
    rows = pooled.loc[
        pooled["geometry_case"].eq(geometry_case)
        & pooled["estimator"].eq(estimator)
        & pooled["shot_budget"].eq(int(shot_budget))
        & pooled["replicate"].eq(int(replicate))
        & pooled["feature_family"].eq(feature_family)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "expected one pooled row for "
            f"{geometry_case}/{estimator}/{shot_budget}/{replicate}/"
            f"{feature_family}; found {len(rows)}"
        )
    return float(rows.iloc[0]["average_precision"])


def _hypothesis_summary(
    pooled: pd.DataFrame,
    shot_summary: pd.DataFrame,
    config: FiniteQuenchAssayConfig,
) -> dict[str, object]:
    baseline = _lookup_ap(
        pooled,
        geometry_case="classical",
        estimator="deterministic",
        shot_budget=0,
        replicate=0,
        feature_family="classical_context",
    )
    primary = _lookup_ap(
        pooled,
        geometry_case=PRIMARY_GEOMETRY,
        estimator="expectation",
        shot_budget=0,
        replicate=0,
        feature_family=PRIMARY_FAMILY,
    )
    final_static = _lookup_ap(
        pooled,
        geometry_case=PRIMARY_GEOMETRY,
        estimator="expectation",
        shot_budget=0,
        replicate=0,
        feature_family="classical_plus_final_static",
    )
    displaced = _lookup_ap(
        pooled,
        geometry_case="reporter_displaced",
        estimator="expectation",
        shot_budget=0,
        replicate=0,
        feature_family=PRIMARY_FAMILY,
    )
    finite = shot_summary.loc[
        shot_summary["geometry_case"].eq(PRIMARY_GEOMETRY)
        & shot_summary["feature_family"].eq(PRIMARY_FAMILY)
        & shot_summary["estimator"].eq("shots")
    ].sort_values("shot_budget")
    finite_rows = finite[
        [
            "shot_budget",
            "replicates",
            "average_precision_mean",
            "average_precision_std",
            "roc_auc_mean",
            "brier_mean",
        ]
    ].to_dict(orient="records")
    return {
        "primary_metric": "pooled_out_of_fold_average_precision",
        "primary_geometry": PRIMARY_GEOMETRY,
        "primary_feature_family": PRIMARY_FAMILY,
        "classical_context_ap": baseline,
        "primary_exact_ap": primary,
        "final_static_exact_ap": final_static,
        "displaced_control_exact_ap": displaced,
        "h1_finite_quench_adds_beyond_final_state": {
            "delta_ap": primary - final_static,
            "supported_descriptively": bool(primary > final_static),
        },
        "h2_finite_quench_adds_beyond_classical": {
            "delta_ap": primary - baseline,
            "supported_descriptively": bool(primary > baseline),
        },
        "h3_undisplaced_reporter_beats_displaced_control": {
            "delta_ap": primary - displaced,
            "supported_descriptively": bool(primary > displaced),
        },
        "h4_uplift_survives_finite_shots": [
            {
                **row,
                "delta_ap_over_classical": (
                    float(row["average_precision_mean"]) - baseline
                ),
                "supported_descriptively": bool(
                    float(row["average_precision_mean"]) > baseline
                ),
            }
            for row in finite_rows
        ],
        "measurement_settings_per_sample_per_geometry": (
            1 + len(config.response_probe_steps)
        ),
        "nominal_shots_per_sample_per_geometry": [
            {
                "shot_budget": int(shots),
                "total_shots": int(
                    shots * (1 + len(config.response_probe_steps))
                ),
            }
            for shots in config.shot_budgets
        ],
    }


def run_rydberg_finite_quench_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    assay: FiniteQuenchAssayConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    run_id: str | None = None,
) -> Path:
    """Stress-test history-prepared states for measurable L5 warning information."""

    assay.validate()
    candidate_features.validate()
    reservoir.validate()
    ladder_geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("the finite-quench assay requires six atoms")
    if reservoir.shots is not None:
        raise ValueError("set reservoir.shots=None; this assay samples explicitly")

    geometry_cases = {
        "reporter_undisplaced": reporter_geometry(
            ladder_geometry,
            displaced=False,
        ),
        "reporter_displaced": reporter_geometry(
            ladder_geometry,
            displaced=True,
        ),
    }
    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "assay": assay.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "geometry_cases": {
                name: geometry.to_dict()
                for name, geometry in geometry_cases.items()
            },
            "primary_model": {
                "geometry": PRIMARY_GEOMETRY,
                "feature_family": PRIMARY_FAMILY,
                "classifier": "l2_logistic_regression",
            },
            "predeclared_hypotheses": {
                "h1": (
                    "the finite-quench reporter improves L5 average precision "
                    "beyond the unquenched final state"
                ),
                "h2": (
                    "the finite-quench reporter improves average precision "
                    "beyond the causal classical context"
                ),
                "h3": (
                    "the naturally central undisplaced reporter outperforms "
                    "the previously displaced reporter control"
                ),
                "h4": (
                    "the finite-quench uplift remains positive at 1000 and "
                    "5000 shots per measurement setting"
                ),
            },
            "measurement_protocol": {
                "settings_per_sample_per_geometry": (
                    1 + len(assay.response_probe_steps)
                ),
                "settings": [
                    "prepared_final_state",
                    *[
                        f"positive_detuning_quench_step_{step}"
                        for step in assay.response_probe_steps
                    ],
                ],
                "derivative_features": False,
                "curvature_features": False,
            },
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(fold_dir)
    available_folds = set(dataset.manifest["fold"].astype(int).unique())
    missing = set(assay.folds).difference(available_folds)
    if missing:
        raise ValueError(
            f"requested folds are absent: {sorted(missing)}; "
            f"available={sorted(available_folds)}"
        )
    level_channel = resolve_level_channel(
        dataset,
        name=assay.level_channel_name,
        fallback=assay.fallback_level_channel,
    )

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []

    for fold in assay.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=(int(assay.lead),),
            max_per_class=assay.max_per_class,
            seed=assay.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("finite-quench assay must not receive test rows")
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        source = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=assay.sequence_length,
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
            raise RuntimeError(f"fold {fold}: empty train or validation rows")
        labels = frame["label"].to_numpy(dtype=int)
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train)
        har_residuals, warning_train = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=assay.prequential_blocks,
        )
        har_context = har.copy()
        har_context[warning_train] = (
            y[warning_train] - har_residuals[warning_train]
        )
        if len(np.unique(labels[warning_train])) != 2:
            raise RuntimeError(
                f"fold {fold}: prequential classifier rows lack both labels"
            )

        raw_sequences = build_candidate_sequences(
            source,
            "level_instability",
            candidate_features,
        )
        scaler = fit_channel_scaler(raw_sequences, train)
        encoded = transform_candidate_sequences(raw_sequences, scaler)
        classical, classical_names = classical_context_features(
            source,
            raw_sequences[:, :, 1],
            har_context,
        )

        baseline_probability = fit_warning_classifier(
            classical,
            labels,
            warning_train,
            validation,
            classifier_c=assay.classifier_c,
            seed=assay.seed + int(fold),
        )
        baseline_diagnostics = feature_diagnostics(
            classical,
            labels,
            warning_train,
        )
        metric_rows.append(
            _metric_row(
                frame,
                baseline_probability,
                validation,
                warning_train,
                fold=int(fold),
                geometry_case="classical",
                estimator="deterministic",
                shot_budget=0,
                replicate=0,
                feature_family="classical_context",
                diagnostics=baseline_diagnostics,
            )
        )
        prediction_frames.append(
            _prediction_frame(
                frame,
                baseline_probability,
                validation,
                fold=int(fold),
                geometry_case="classical",
                estimator="deterministic",
                shot_budget=0,
                replicate=0,
                feature_family="classical_context",
            )
        )

        for geometry_index, (case_name, geometry) in enumerate(
            geometry_cases.items()
        ):
            prepared = prepare_ladder_history(
                encoded,
                reservoir,
                geometry,
                interaction_scale=assay.interaction_scale,
            )
            quench_states = evolve_finite_quench(
                prepared,
                reservoir,
                delta_offset_rad_us=assay.quench_delta_offset_rad_us,
                response_probe_steps=assay.response_probe_steps,
            )
            estimator_specs: list[tuple[str, int, int]] = [
                ("expectation", 0, 0)
            ]
            estimator_specs.extend(
                ("shots", int(shots), int(replicate))
                for shots in assay.shot_budgets
                for replicate in range(1, assay.shot_replicates + 1)
            )
            for estimator, shot_budget, replicate in estimator_specs:
                measurement_seed = int(
                    (
                        assay.seed
                        + 1_000_003 * int(fold)
                        + 100_003 * geometry_index
                        + 101 * int(shot_budget)
                        + 10_007 * int(replicate)
                    )
                    % (2**32 - 1)
                )
                blocks, block_names = finite_quench_feature_blocks(
                    prepared,
                    quench_states,
                    geometry,
                    response_probe_steps=assay.response_probe_steps,
                    shots=None if estimator == "expectation" else shot_budget,
                    seed=measurement_seed,
                )
                matrices, names = finite_quench_feature_families(
                    classical,
                    classical_names,
                    blocks,
                    block_names,
                )
                for family, matrix in matrices.items():
                    probability = fit_warning_classifier(
                        matrix,
                        labels,
                        warning_train,
                        validation,
                        classifier_c=assay.classifier_c,
                        seed=measurement_seed,
                    )
                    diagnostics = feature_diagnostics(
                        matrix,
                        labels,
                        warning_train,
                    )
                    metric_rows.append(
                        _metric_row(
                            frame,
                            probability,
                            validation,
                            warning_train,
                            fold=int(fold),
                            geometry_case=case_name,
                            estimator=estimator,
                            shot_budget=shot_budget,
                            replicate=replicate,
                            feature_family=family,
                            diagnostics=diagnostics,
                        )
                    )
                    prediction_frames.append(
                        _prediction_frame(
                            frame,
                            probability,
                            validation,
                            fold=int(fold),
                            geometry_case=case_name,
                            estimator=estimator,
                            shot_budget=shot_budget,
                            replicate=replicate,
                            feature_family=family,
                        )
                    )
                elementary = {
                    key: blocks[key]
                    for key in (
                        "final_static",
                        "quench_modes_raw",
                        "quench_modes_delta",
                        "reporter_raw",
                        "reporter_delta",
                    )
                }
                elementary_names = {
                    key: block_names[key] for key in elementary
                }
                feature_frames.append(
                    long_feature_frame(
                        frame,
                        elementary,
                        elementary_names,
                        fold=int(fold),
                        geometry_case=case_name,
                        estimator=estimator,
                        shot_budget=shot_budget,
                        replicate=replicate,
                    )
                )

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    feature_values = pd.concat(feature_frames, ignore_index=True)
    if predictions["sample_id"].isna().any():
        raise RuntimeError("prediction output contains missing sample IDs")
    if not feature_values["fold_split"].isin(["train", "val"]).all():
        raise RuntimeError("finite-quench feature output contains non-development rows")

    pooled_metrics = _pooled_metrics(predictions)
    shot_summary = _shot_summary(pooled_metrics)
    geometry_effect = _geometry_effect(pooled_metrics)
    feature_fidelity = _feature_fidelity(feature_values)
    hypotheses = _hypothesis_summary(pooled_metrics, shot_summary, assay)

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    feature_values.to_csv(
        run_dir / "feature_values.csv.gz",
        index=False,
        compression="gzip",
    )
    pooled_metrics.to_csv(run_dir / "pooled_metrics.csv", index=False)
    shot_summary.to_csv(run_dir / "shot_summary.csv", index=False)
    geometry_effect.to_csv(run_dir / "geometry_effect.csv", index=False)
    feature_fidelity.to_csv(run_dir / "feature_fidelity.csv", index=False)
    (run_dir / "hypothesis_summary.json").write_text(
        json.dumps(hypotheses, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exact = pooled_metrics.loc[
        pooled_metrics["estimator"].isin(["deterministic", "expectation"])
        & pooled_metrics["shot_budget"].eq(0)
    ].sort_values("average_precision", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    plot = exact.head(16).copy()
    labels_plot = (
        plot["geometry_case"].astype(str)
        + "\n"
        + plot["feature_family"].astype(str)
    )
    ax.bar(labels_plot, plot["average_precision"])
    ax.set_ylabel("Pooled out-of-fold average precision")
    ax.set_title("Exact finite-quench L5 warning comparison")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(run_dir / "exact_finite_quench_auprc.png", dpi=180)
    plt.close(fig)

    primary_shots = shot_summary.loc[
        shot_summary["geometry_case"].eq(PRIMARY_GEOMETRY)
        & shot_summary["feature_family"].eq(PRIMARY_FAMILY)
        & shot_summary["estimator"].eq("shots")
    ].sort_values("shot_budget")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if not primary_shots.empty:
        ax.errorbar(
            primary_shots["shot_budget"],
            primary_shots["average_precision_mean"],
            yerr=primary_shots["average_precision_std"],
            marker="o",
        )
    baseline_ap = hypotheses["classical_context_ap"]
    ax.axhline(float(baseline_ap), linestyle="--", label="Classical context")
    ax.set_xlabel("Shots per measurement setting")
    ax.set_ylabel("Average precision")
    ax.set_title("Finite-quench reporter shot robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "finite_quench_shot_auprc.png", dpi=180)
    plt.close(fig)

    primary_fidelity = feature_fidelity.loc[
        feature_fidelity["geometry_case"].eq(PRIMARY_GEOMETRY)
        & feature_fidelity["feature_family"].isin(
            ["final_static", "reporter_raw", "reporter_delta"]
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for family, local in primary_fidelity.groupby("feature_family"):
        summary = (
            local.groupby("shot_budget")["correlation_with_expectation"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            summary["shot_budget"],
            summary["mean"],
            yerr=summary["std"].fillna(0.0),
            marker="o",
            label=family,
        )
    ax.set_xlabel("Shots per measurement setting")
    ax.set_ylabel("Correlation with exact feature values")
    ax.set_title("Finite-quench feature fidelity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "finite_quench_feature_fidelity.png", dpi=180)
    plt.close(fig)

    summary = {
        "schema_version": 1,
        "status": "development_finite_quench_reporter_assay",
        "evaluation_split": "validation",
        "folds": list(assay.folds),
        "lead": int(assay.lead),
        "test_rows_used": 0,
        "common_test_block_evaluated": False,
        "primary_geometry": PRIMARY_GEOMETRY,
        "primary_feature_family": PRIMARY_FAMILY,
        "quench": {
            "control": "global_detuning",
            "offset_rad_us": float(assay.quench_delta_offset_rad_us),
            "response_steps": list(assay.response_probe_steps),
            "derivative_features": False,
            "curvature_features": False,
        },
        "measurement_settings_per_sample_per_geometry": (
            1 + len(assay.response_probe_steps)
        ),
        "shot_budgets": list(assay.shot_budgets),
        "shot_replicates": int(assay.shot_replicates),
        "hypothesis_summary": hypotheses,
        "known_limitations": [
            "All eight rolling validation folds are development data.",
            "The common held-out test block remains unopened.",
            "The quench amplitude and response times are one predeclared design, not an optimized phase diagram.",
            "Finite-shot simulations include statistical sampling but not hardware noise.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir
