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
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.rydberg_susceptibility_tools import (
    SusceptibilityAssayConfig,
    classical_context_features,
    feature_diagnostics,
    feature_families,
    fit_warning_classifier,
    long_feature_frame,
    prepare_ladder_history,
    probe_prepared_ladder,
    reporter_geometry,
    susceptibility_feature_blocks,
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
        "train_rows": int(frame["fold_split"].eq("train").sum()),
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
        mask = np.ones(len(local), dtype=bool)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "folds": int(local["fold"].nunique()),
                "samples": int(len(local)),
                **warning_metrics(labels, probabilities, mask),
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


def _impurity_effect(pooled: pd.DataFrame) -> pd.DataFrame:
    qrc = pooled.loc[
        pooled["geometry_case"].isin(
            ["reporter_displaced", "reporter_undisplaced"]
        )
    ].copy()
    keys = ["estimator", "shot_budget", "replicate", "feature_family"]
    left = qrc.loc[
        qrc["geometry_case"].eq("reporter_displaced")
    ].set_index(keys)
    right = qrc.loc[
        qrc["geometry_case"].eq("reporter_undisplaced")
    ].set_index(keys)
    joined = left.join(
        right,
        how="inner",
        lsuffix="_displaced",
        rsuffix="_undisplaced",
    ).reset_index()
    if joined.empty:
        return joined
    joined["delta_average_precision_displaced_minus_undisplaced"] = (
        joined["average_precision_displaced"]
        - joined["average_precision_undisplaced"]
    )
    joined["delta_roc_auc_displaced_minus_undisplaced"] = (
        joined["roc_auc_displaced"] - joined["roc_auc_undisplaced"]
    )
    joined["delta_brier_displaced_minus_undisplaced"] = (
        joined["brier_displaced"] - joined["brier_undisplaced"]
    )
    return joined


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
            "expected one pooled metric row for "
            f"{geometry_case}/{estimator}/{shot_budget}/{replicate}/"
            f"{feature_family}, found {len(rows)}"
        )
    return float(rows.iloc[0]["average_precision"])


def _hypothesis_summary(
    pooled: pd.DataFrame,
    shot_summary: pd.DataFrame,
) -> dict[str, object]:
    baseline = _lookup_ap(
        pooled,
        geometry_case="classical",
        estimator="deterministic",
        shot_budget=0,
        replicate=0,
        feature_family="classical_context",
    )
    static = _lookup_ap(
        pooled,
        geometry_case="reporter_displaced",
        estimator="expectation",
        shot_budget=0,
        replicate=0,
        feature_family="classical_plus_static",
    )
    susceptibility = _lookup_ap(
        pooled,
        geometry_case="reporter_displaced",
        estimator="expectation",
        shot_budget=0,
        replicate=0,
        feature_family="classical_plus_susceptibility",
    )
    reporter = _lookup_ap(
        pooled,
        geometry_case="reporter_displaced",
        estimator="expectation",
        shot_budget=0,
        replicate=0,
        feature_family="classical_plus_reporter",
    )
    reporter_without_displacement = _lookup_ap(
        pooled,
        geometry_case="reporter_undisplaced",
        estimator="expectation",
        shot_budget=0,
        replicate=0,
        feature_family="classical_plus_reporter",
    )
    payload: dict[str, object] = {
        "primary_metric": "pooled_out_of_fold_average_precision",
        "classical_context_ap": baseline,
        "classical_plus_static_ap": static,
        "classical_plus_susceptibility_ap": susceptibility,
        "classical_plus_reporter_ap": reporter,
        "undisplaced_classical_plus_reporter_ap": reporter_without_displacement,
        "h1_susceptibility_adds_beyond_static": {
            "delta_ap": susceptibility - static,
            "supported_descriptively": bool(susceptibility > static),
        },
        "h2_reporter_displacement_adds_information": {
            "delta_ap": reporter - reporter_without_displacement,
            "supported_descriptively": bool(
                reporter > reporter_without_displacement
            ),
        },
        "h3_susceptibility_adds_beyond_classical": {
            "delta_ap": susceptibility - baseline,
            "supported_descriptively": bool(susceptibility > baseline),
        },
    }
    finite = shot_summary.loc[
        shot_summary["geometry_case"].eq("reporter_displaced")
        & shot_summary["feature_family"].eq(
            "classical_plus_susceptibility"
        )
        & shot_summary["estimator"].eq("shots")
    ].sort_values("shot_budget")
    payload["finite_shot_primary"] = finite[
        [
            "shot_budget",
            "replicates",
            "average_precision_mean",
            "average_precision_std",
        ]
    ].to_dict(orient="records")
    return payload


def run_rydberg_susceptibility_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    assay: SusceptibilityAssayConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    run_id: str | None = None,
) -> Path:
    """Interrogate history-prepared states for L5 transition susceptibility."""

    assay.validate()
    candidate_features.validate()
    reservoir.validate()
    ladder_geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("the susceptibility assay requires six atoms")
    if reservoir.shots is not None:
        raise ValueError("set reservoir.shots=None; this assay samples explicitly")

    geometry_cases = {
        "reporter_displaced": reporter_geometry(
            ladder_geometry,
            displaced=True,
        ),
        "reporter_undisplaced": reporter_geometry(
            ladder_geometry,
            displaced=False,
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
            "predeclared_hypotheses": {
                "h1": (
                    "susceptibility features improve L5 transition-control "
                    "average precision beyond the unprobed state"
                ),
                "h2": (
                    "the displaced reporter site improves reporter-feature "
                    "average precision relative to the same undisplaced site"
                ),
                "h3": (
                    "the susceptibility uplift remains positive under finite "
                    "bitstring sampling"
                ),
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
            raise RuntimeError("susceptibility assay must not receive test rows")
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
            har,
        )
        baseline_probability = fit_warning_classifier(
            classical,
            labels,
            train,
            validation,
            classifier_c=assay.classifier_c,
            seed=assay.seed + int(fold),
        )
        baseline_diagnostics = feature_diagnostics(
            classical,
            labels,
            train,
        )
        metric_rows.append(
            _metric_row(
                frame,
                baseline_probability,
                validation,
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
            branches = probe_prepared_ladder(
                prepared,
                reservoir,
                delta_offset_rad_us=assay.probe_delta_offset_rad_us,
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
                blocks, block_names = susceptibility_feature_blocks(
                    prepared,
                    branches,
                    geometry,
                    delta_offset_rad_us=assay.probe_delta_offset_rad_us,
                    response_probe_steps=assay.response_probe_steps,
                    shots=None if estimator == "expectation" else shot_budget,
                    seed=measurement_seed,
                )
                matrices, names = feature_families(
                    classical,
                    classical_names,
                    blocks,
                    block_names,
                )
                for family, matrix in matrices.items():
                    probability = fit_warning_classifier(
                        matrix,
                        labels,
                        train,
                        validation,
                        classifier_c=assay.classifier_c,
                        seed=measurement_seed,
                    )
                    diagnostics = feature_diagnostics(
                        matrix,
                        labels,
                        train,
                    )
                    metric_rows.append(
                        _metric_row(
                            frame,
                            probability,
                            validation,
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
                        "static_modes",
                        "chi_modes",
                        "kappa_modes",
                        "reporter_chi",
                        "reporter_kappa",
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
    pooled_metrics = _pooled_metrics(predictions)
    shot_summary = _shot_summary(pooled_metrics)
    impurity_effect = _impurity_effect(pooled_metrics)
    hypotheses = _hypothesis_summary(pooled_metrics, shot_summary)

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
    impurity_effect.to_csv(run_dir / "impurity_effect.csv", index=False)
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
    labels = (
        plot["geometry_case"].astype(str)
        + "\n"
        + plot["feature_family"].astype(str)
    )
    ax.bar(labels, plot["average_precision"])
    ax.axhline(
        float(
            exact.loc[
                exact["feature_family"].eq("classical_context"),
                "average_precision",
            ].iloc[0]
        ),
        linestyle="--",
        linewidth=1.0,
        label="classical context",
    )
    ax.set_ylabel("Pooled L5 average precision")
    ax.set_title("Exact-state susceptibility and reporter features")
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exact_auprc.png", dpi=180)
    plt.close(fig)

    primary = shot_summary.loc[
        shot_summary["geometry_case"].eq("reporter_displaced")
        & shot_summary["feature_family"].isin(
            [
                "classical_plus_static",
                "classical_plus_reporter",
                "classical_plus_susceptibility",
            ]
        )
    ].copy()
    expectation_primary = pooled_metrics.loc[
        pooled_metrics["geometry_case"].eq("reporter_displaced")
        & pooled_metrics["estimator"].eq("expectation")
        & pooled_metrics["feature_family"].isin(
            primary["feature_family"].unique()
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for family in sorted(primary["feature_family"].unique()):
        finite = primary.loc[
            primary["feature_family"].eq(family)
            & primary["estimator"].eq("shots")
        ].sort_values("shot_budget")
        exact_row = expectation_primary.loc[
            expectation_primary["feature_family"].eq(family)
        ]
        x = [0] + finite["shot_budget"].astype(int).tolist()
        y = [float(exact_row["average_precision"].iloc[0])] + finite[
            "average_precision_mean"
        ].tolist()
        error = [0.0] + finite["average_precision_std"].tolist()
        ax.errorbar(x, y, yerr=error, marker="o", label=family)
    ax.set_xlabel("Shots per branch and probe (0 = exact expectation)")
    ax.set_ylabel("Pooled L5 average precision")
    ax.set_title("Finite-shot survival of susceptibility features")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "shot_auprc.png", dpi=180)
    plt.close(fig)

    reporter_effect = impurity_effect.loc[
        impurity_effect["estimator"].eq("expectation")
        & impurity_effect["shot_budget"].eq(0)
        & impurity_effect["feature_family"].str.contains("reporter")
    ].sort_values(
        "delta_average_precision_displaced_minus_undisplaced",
        ascending=False,
    )
    if not reporter_effect.empty:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.bar(
            reporter_effect["feature_family"],
            reporter_effect[
                "delta_average_precision_displaced_minus_undisplaced"
            ],
        )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylabel("Delta average precision")
        ax.set_title("Value of the displaced reporter site")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(run_dir / "reporter_impurity_effect.png", dpi=180)
        plt.close(fig)

    summary = {
        "schema_version": 1,
        "status": "development_rydberg_susceptibility_assay",
        "target": "L5 transition occurrence versus matched L5 controls",
        "primary_metric": "pooled out-of-fold average precision",
        "development_folds": list(assay.folds),
        "evaluation_split": "validation",
        "test_rows_used": 0,
        "common_test_block_evaluated": False,
        "physical_experiment": {
            "history_preparation": (
                "persistent 40-step level-plus-instability evolution"
            ),
            "interrogation": (
                "central detuning probe around the final history-conditioned "
                "operating point"
            ),
            "probe_offset_rad_us": assay.probe_delta_offset_rad_us,
            "response_probe_steps": list(assay.response_probe_steps),
            "reporter_site": int(ladder_geometry.defect_site),
            "reporter_control": (
                "same staggered ladder with reporter displacement removed"
            ),
        },
        "classifier": {
            "type": "fixed L2 logistic regression",
            "C": assay.classifier_c,
            "selection": "none",
        },
        "shot_budgets": [0, *assay.shot_budgets],
        "shot_replicates": assay.shot_replicates,
        "hypothesis_summary": hypotheses,
        "known_limitations": [
            "All eight rolling validation folds are development data.",
            "The common final test block remains untouched.",
            "The probe amplitude and reporter definition are theory-driven but not independently optimized.",
            "Finite-shot estimates include sampling noise but not hardware noise.",
            "This assay tests transition separability, not the ten-day volatility path.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir
