from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.forecast_warning_tools import (
    MECHANISM_SCORE,
    PRIMARY_SCORE,
    FrozenPathWarningConfig,
    build_path_score_frame,
    cluster_bootstrap_ap,
    horizon_curve_table,
    load_prediction_source,
    metric_table,
    paired_cluster_bootstrap_ap,
)


SCORE_COLUMNS = (
    "onset_rise",
    "peak_rise",
    "terminal_rise",
    "path_slope",
    "onset_level",
    "peak_level",
    "forecast_mean",
    "qrc_onset_correction",
    "qrc_peak_correction",
    "qrc_mean_correction",
    "qrc_correction_slope",
    "oracle_onset_rise",
)


def _metric_lookup(
    metrics: pd.DataFrame,
    *,
    period: str,
    model: str,
    score: str,
    metric: str = "average_precision",
) -> float:
    rows = metrics.loc[
        metrics["period"].eq(period)
        & metrics["fold"].eq(0)
        & metrics["model"].eq(model)
        & metrics["score"].eq(score)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one metric row for {period}/{model}/{score}, found {len(rows)}"
        )
    return float(rows.iloc[0][metric])


def _summary_payload(
    metrics: pd.DataFrame,
    bootstraps: list[dict[str, object]],
    config: FrozenPathWarningConfig,
) -> dict[str, object]:
    periods = ("all_folds", "confirmation_folds_4_8")
    primary = {}
    mechanism = {}
    for period in periods:
        har_ap = _metric_lookup(
            metrics,
            period=period,
            model="har",
            score=PRIMARY_SCORE,
        )
        ladder_ap = _metric_lookup(
            metrics,
            period=period,
            model="frozen_ladder_symmetric_1p25",
            score=PRIMARY_SCORE,
        )
        chain_ap = _metric_lookup(
            metrics,
            period=period,
            model="chain_interacting_1p00",
            score=PRIMARY_SCORE,
        )
        off_ap = _metric_lookup(
            metrics,
            period=period,
            model="chain_interaction_off",
            score=PRIMARY_SCORE,
        )
        primary[period] = {
            "har_ap": har_ap,
            "ladder_ap": ladder_ap,
            "interacting_chain_ap": chain_ap,
            "interaction_off_ap": off_ap,
            "ladder_minus_har_ap": ladder_ap - har_ap,
        }

        ladder_correction_ap = _metric_lookup(
            metrics,
            period=period,
            model="frozen_ladder_symmetric_1p25",
            score=MECHANISM_SCORE,
        )
        chain_correction_ap = _metric_lookup(
            metrics,
            period=period,
            model="chain_interacting_1p00",
            score=MECHANISM_SCORE,
        )
        off_correction_ap = _metric_lookup(
            metrics,
            period=period,
            model="chain_interaction_off",
            score=MECHANISM_SCORE,
        )
        mechanism[period] = {
            "ladder_mean_correction_ap": ladder_correction_ap,
            "interacting_chain_mean_correction_ap": chain_correction_ap,
            "interaction_off_mean_correction_ap": off_correction_ap,
            "ladder_above_matched_prevalence": ladder_correction_ap - 0.5,
            "interacting_chain_above_matched_prevalence": (
                chain_correction_ap - 0.5
            ),
        }

    return {
        "schema_version": 1,
        "status": "development_forecast_derived_transition_warning_audit",
        "evaluation_split": "validation",
        "test_rows_used": 0,
        "common_test_block_evaluated": False,
        "lead": int(config.lead),
        "baseline_horizon": int(config.baseline_horizon),
        "onset_horizon": int(config.onset_horizon),
        "primary_score": {
            "name": PRIMARY_SCORE,
            "definition": "forecast_h5_minus_forecast_h1",
            "interpretation": (
                "predicted rise from the first forecast day to the known L5 "
                "transition-onset horizon"
            ),
        },
        "mechanism_score": {
            "name": MECHANISM_SCORE,
            "definition": "mean_over_h1_to_h10_of_qrc_forecast_minus_har_forecast",
            "interpretation": "average QRC residual correction over the forecast path",
        },
        "primary_results": primary,
        "mechanism_results": mechanism,
        "bootstrap_results": bootstraps,
        "predeclared_questions": {
            "h1": (
                "Does the frozen ladder forecast-onset rise separate L5 transitions "
                "from matched controls better than HAR?"
            ),
            "h2": (
                "Does the QRC correction itself contain transition-occurrence "
                "information above the matched prevalence baseline?"
            ),
            "h3": (
                "Do interacting reservoirs provide stronger warning scores than the "
                "interaction-off temporal reservoir?"
            ),
        },
        "known_limitations": [
            "All eight rolling folds are development or validation data.",
            "The common final test block remains unopened.",
            "The L5 transition onset is known only for retrospective evaluation.",
            "The score uses only the univariate volatility-path forecast.",
            "Matched controls make the prevalence approximately one half and do not represent live-market prevalence.",
        ],
    }


def run_frozen_path_warning_assay(
    *,
    development_run_dir: Path,
    confirmation_run_dir: Path,
    results_root: Path,
    config: FrozenPathWarningConfig,
    run_id: str | None = None,
) -> Path:
    """Evaluate transition-warning information already present in frozen paths."""

    config.validate()
    run_dir = begin_run(
        results_root,
        {
            "development_run_dir": str(development_run_dir),
            "confirmation_run_dir": str(confirmation_run_dir),
            "config": config.to_dict(),
            "model_sources": {
                "folds_1_3": {
                    "har": "har",
                    "chain_interaction_off": "reference_chain_interaction_off",
                    "chain_interacting_1p00": "reference_chain_interacting_1p00",
                    "frozen_ladder_symmetric_1p25": (
                        "candidate_ladder_ordered_1p25_symmetric_modes"
                    ),
                },
                "folds_4_8": {
                    "har": "har",
                    "chain_interaction_off": "chain_interaction_off",
                    "chain_interacting_1p00": "chain_interacting_1p00",
                    "frozen_ladder_symmetric_1p25": (
                        "frozen_ladder_symmetric_1p25"
                    ),
                },
            },
            "primary_score": PRIMARY_SCORE,
            "mechanism_score": MECHANISM_SCORE,
            "test_rows_allowed": False,
            "reservoir_resimulation": False,
        },
        run_id=run_id,
    )

    development = load_prediction_source(
        development_run_dir,
        source="development",
        config=config,
    )
    confirmation = load_prediction_source(
        confirmation_run_dir,
        source="confirmation",
        config=config,
    )
    predictions = pd.concat([development, confirmation], ignore_index=True)
    scores = build_path_score_frame(predictions, config)
    if not scores["fold"].isin(config.all_folds).all():
        raise RuntimeError("warning audit contains unexpected folds")

    metrics = metric_table(scores, config, score_columns=SCORE_COLUMNS)
    horizon_curves = horizon_curve_table(scores, config)

    bootstrap_results = [
        paired_cluster_bootstrap_ap(
            scores,
            model_a="frozen_ladder_symmetric_1p25",
            model_b="har",
            score_column=PRIMARY_SCORE,
            folds=folds,
            replicates=config.bootstrap_replicates,
            seed=config.seed + index,
        )
        for index, folds in enumerate(
            (config.all_folds, config.confirmation_folds),
            start=1,
        )
    ]
    bootstrap_results.extend(
        [
            paired_cluster_bootstrap_ap(
                scores,
                model_a="chain_interacting_1p00",
                model_b="chain_interaction_off",
                score_column=MECHANISM_SCORE,
                folds=folds,
                replicates=config.bootstrap_replicates,
                seed=config.seed + 100 + index,
            )
            for index, folds in enumerate(
                (config.all_folds, config.confirmation_folds),
                start=1,
            )
        ]
    )
    bootstrap_results.extend(
        [
            cluster_bootstrap_ap(
                scores,
                model=model,
                score_column=MECHANISM_SCORE,
                folds=config.confirmation_folds,
                replicates=config.bootstrap_replicates,
                seed=config.seed + 200 + index,
            )
            for index, model in enumerate(
                (
                    "chain_interaction_off",
                    "chain_interacting_1p00",
                    "frozen_ladder_symmetric_1p25",
                ),
                start=1,
            )
        ]
    )

    summary = _summary_payload(metrics, bootstrap_results, config)

    predictions.to_csv(
        run_dir / "selected_prediction_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    scores.to_csv(run_dir / "path_warning_scores.csv", index=False)
    metrics.to_csv(run_dir / "warning_score_metrics.csv", index=False)
    horizon_curves.to_csv(run_dir / "warning_horizon_curves.csv", index=False)
    (run_dir / "bootstrap_results.json").write_text(
        json.dumps(bootstrap_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pooled_primary = metrics.loc[
        metrics["period"].eq("all_folds")
        & metrics["fold"].eq(0)
        & metrics["score"].eq(PRIMARY_SCORE)
    ].sort_values("average_precision", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(pooled_primary["model"], pooled_primary["average_precision"])
    ax.axhline(0.5, linestyle="--", label="Matched prevalence")
    ax.set_ylabel("Average precision")
    ax.set_title("L5 transition warning from predicted onset rise")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "onset_rise_average_precision.png", dpi=180)
    plt.close(fig)

    confirmation_curve = horizon_curves.loc[
        horizon_curves["period"].eq("confirmation_folds_4_8")
        & horizon_curves["score_family"].eq("forecast_rise_from_h1")
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model, local in confirmation_curve.groupby("model"):
        local = local.sort_values("horizon")
        ax.plot(
            local["horizon"],
            local["average_precision"],
            marker="o",
            label=model,
        )
    ax.axhline(0.5, linestyle="--")
    ax.axvline(config.onset_horizon, linestyle=":", label="L5 onset horizon")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Average precision")
    ax.set_title("Confirmation-fold warning information across the forecast path")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "confirmation_warning_horizon_curve.png", dpi=180)
    plt.close(fig)

    mechanism = metrics.loc[
        metrics["period"].eq("confirmation_folds_4_8")
        & metrics["fold"].eq(0)
        & metrics["score"].eq(MECHANISM_SCORE)
    ].sort_values("average_precision", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(mechanism["model"], mechanism["average_precision"])
    ax.axhline(0.5, linestyle="--", label="Matched prevalence")
    ax.set_ylabel("Average precision")
    ax.set_title("Transition information in the mean QRC correction")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "confirmation_qrc_correction_average_precision.png", dpi=180)
    plt.close(fig)

    return run_dir
