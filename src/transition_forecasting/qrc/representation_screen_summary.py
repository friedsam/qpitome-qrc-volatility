from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def summarize_representation_screen(run_dir: Path) -> tuple[Path, Path]:
    """Rebuild the representation summary using only configurations in all folds."""
    run_dir = Path(run_dir)
    metrics = pd.read_csv(run_dir / "model_metrics.csv")
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    expected_folds = len(params["screen"]["folds"])
    requested_full_rank = 0 in params["screen"]["components"]

    candidates = metrics.loc[~metrics["model_family"].eq("har")].copy()
    candidates["component_spec"] = candidates["components"].astype(str)
    if requested_full_rank:
        maximum = candidates.groupby(
            ["fold", "representation", "model_family", "seed"]
        )["components"].transform("max")
        candidates.loc[
            candidates["components"].eq(maximum), "component_spec"
        ] = "full"

    fold_summary = (
        candidates.groupby(
            [
                "fold",
                "representation",
                "model_family",
                "readout_mode",
                "component_spec",
                "alpha",
            ],
            as_index=False,
        )
        .agg(
            val_qlike=("val_qlike", "mean"),
            val_rmse=("val_rmse", "mean"),
            val_mz_r2=("val_mz_r2", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    summary = (
        fold_summary.groupby(
            [
                "representation",
                "model_family",
                "readout_mode",
                "component_spec",
                "alpha",
            ],
            as_index=False,
        )
        .agg(
            mean_val_qlike=("val_qlike", "mean"),
            sd_val_qlike=("val_qlike", "std"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_mz_r2=("val_mz_r2", "mean"),
            folds=("fold", "nunique"),
            min_seeds=("seeds", "min"),
        )
    )
    summary = summary.loc[summary["folds"].eq(expected_folds)].copy()
    best = (
        summary.sort_values(
            [
                "representation",
                "model_family",
                "mean_val_qlike",
                "mean_val_rmse",
            ]
        )
        .groupby(["representation", "model_family"], as_index=False)
        .head(1)
    )
    csv_path = run_dir / "best_mean_configuration_corrected.csv"
    best.to_csv(csv_path, index=False)

    pivot = best.pivot(
        index="representation",
        columns="model_family",
        values="mean_val_qlike",
    )
    ax = pivot.plot(kind="bar", figsize=(11, 5.8))
    har_mean = metrics.loc[
        metrics["model_family"].eq("har"), "val_qlike"
    ].mean()
    ax.axhline(har_mean, linestyle="--", linewidth=1.2, label="HAR mean")
    ax.set_ylabel("Mean validation QLIKE")
    ax.set_xlabel("Input representation")
    ax.set_title("Representation screen: all-fold comparison")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Model family", fontsize=8)
    ax.figure.tight_layout()
    plot_path = run_dir / "representation_screen_qlike_corrected.png"
    ax.figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)
    return csv_path, plot_path
