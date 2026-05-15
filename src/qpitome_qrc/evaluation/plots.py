from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str],
    out_path: str | Path,
    title: str = "Feature correlation heatmap",
) -> Path:
    """Save absolute correlation heatmap for selected features."""
    out_path = Path(out_path)
    corr = df[features].corr().abs()

    fig, ax = plt.subplots(figsize=(max(8, len(features) * 0.45), max(6, len(features) * 0.45)))
    im = ax.imshow(corr, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="absolute correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_metric_barplot(
    metrics: pd.DataFrame,
    metric_col: str,
    out_path: str | Path,
    title: str | None = None,
    top_n: int = 12,
) -> Path:
    """Save horizontal barplot of top models by metric."""
    out_path = Path(out_path)
    data = metrics.sort_values(metric_col, ascending=True).tail(top_n).copy()
    labels = data["feature_set"] + " :: " + data["model_name"]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(data))))
    ax.barh(labels, data[metric_col])
    ax.set_xlabel(metric_col)
    ax.set_title(title or f"Top models by {metric_col}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pr_roc_curves(
    result_items: Mapping[str, object],
    y_val: np.ndarray,
    out_pr_path: str | Path,
    out_roc_path: str | Path,
    max_curves: int = 8,
) -> tuple[Path, Path]:
    """Save validation PR and ROC curves for selected ClassicalRunResult objects."""
    out_pr_path = Path(out_pr_path)
    out_roc_path = Path(out_roc_path)
    items = list(result_items.items())[:max_curves]

    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))

    for label, result in items:
        PrecisionRecallDisplay.from_predictions(y_val, result.val_scores, name=label, ax=ax_pr)
        RocCurveDisplay.from_predictions(y_val, result.val_scores, name=label, ax=ax_roc)

    ax_pr.set_title("Validation precision-recall curves")
    ax_pr.grid(alpha=0.25)
    ax_roc.set_title("Validation ROC curves")
    ax_roc.grid(alpha=0.25)

    fig_pr.tight_layout()
    fig_roc.tight_layout()
    fig_pr.savefig(out_pr_path, dpi=180, bbox_inches="tight")
    fig_roc.savefig(out_roc_path, dpi=180, bbox_inches="tight")
    plt.close(fig_pr)
    plt.close(fig_roc)
    return out_pr_path, out_roc_path


def save_pca_variance_plot(pca_table: pd.DataFrame, out_path: str | Path) -> Path:
    """Save PCA cumulative explained variance plot."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        pca_table["component"],
        pca_table["cumulative_explained_variance"],
        marker="o",
        linewidth=1,
    )
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title("PCA cumulative explained variance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_feature_importance_plot(
    importance: pd.DataFrame,
    out_path: str | Path,
    title: str = "Permutation feature importance",
    top_n: int = 15,
) -> Path:
    """Save horizontal feature-importance plot."""
    out_path = Path(out_path)
    data = importance.sort_values("importance_mean", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(data))))
    ax.barh(data["feature"], data["importance_mean"], xerr=data["importance_std"])
    ax.set_xlabel("Permutation importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path
