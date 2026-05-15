from __future__ import annotations

from pathlib import Path

import pandas as pd

from qpitome_qrc.baselines.classical import (
    default_model_configs,
    permutation_importance_table,
    run_classical_suite,
)
from qpitome_qrc.baselines.esn import COMPACT_FEATURES, EXPANDED_FEATURES, TARGET
from qpitome_qrc.data.loaders import load_market_stress_data
from qpitome_qrc.data.splits import chronological_tabular_split, describe_splits, split_arrays
from qpitome_qrc.dimred.feature_select import CorrelationPruner, correlation_summary
from qpitome_qrc.dimred.reducers import DimReductionConfig, build_dimred_pipeline, pca_explained_variance_table
from qpitome_qrc.evaluation.plots import (
    ensure_dir,
    save_correlation_heatmap,
    save_feature_importance_plot,
    save_metric_barplot,
    save_pca_variance_plot,
    save_pr_roc_curves,
)
from qpitome_qrc.evaluation.reports import build_pdf_report


REPORT_DIR = Path("reports/classical_baselines")
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


def available(features: list[str], columns: list[str]) -> list[str]:
    """Keep only features present in the dataset."""
    return [f for f in features if f in columns]


def add_corr_pruned_feature_set(
    arrays_by_set: dict,
    feature_names_by_set: dict,
    splits: dict[str, pd.DataFrame],
    base_features: list[str],
    target: str,
    threshold: float = 0.95,
) -> None:
    """Add correlation-pruned feature set fit on train only."""
    pruner = CorrelationPruner(threshold=threshold, feature_names=base_features)
    X_train = splits["train"][base_features]
    pruner.fit(X_train)

    arrays_by_set[f"compact_corr_pruned_{threshold}"] = {}
    for split_name, split in splits.items():
        X = pruner.transform(split[base_features])
        y = split[target].to_numpy(dtype=int)
        arrays_by_set[f"compact_corr_pruned_{threshold}"][split_name] = (X, y)

    feature_names_by_set[f"compact_corr_pruned_{threshold}"] = list(pruner.get_feature_names_out())


def add_pca_feature_set(
    arrays_by_set: dict,
    feature_names_by_set: dict,
    splits: dict[str, pd.DataFrame],
    base_features: list[str],
    target: str,
    n_components: int,
) -> pd.DataFrame:
    """Add PCA-reduced feature set fit on train only."""
    config = DimReductionConfig(kind="pca", n_components=n_components, scale=True)
    pipe = build_dimred_pipeline(config)

    X_train = splits["train"][base_features].to_numpy(dtype=float)
    pipe.fit(X_train)

    name = f"compact_pca_{n_components}"
    arrays_by_set[name] = {}
    for split_name, split in splits.items():
        X = pipe.transform(split[base_features].to_numpy(dtype=float))
        y = split[target].to_numpy(dtype=int)
        arrays_by_set[name][split_name] = (X, y)

    feature_names_by_set[name] = [f"PC{i + 1}" for i in range(n_components)]
    return pca_explained_variance_table(pipe)


def main() -> None:
    ensure_dir(REPORT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(TABLE_DIR)

    df = load_market_stress_data()
    compact = available(COMPACT_FEATURES, list(df.columns))
    expanded = available(EXPANDED_FEATURES, list(df.columns))

    splits = chronological_tabular_split(df)
    split_summary = describe_splits(splits, TARGET)

    dataset_summary = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "date_min": df["date"].min(),
                "date_max": df["date"].max(),
                "target": TARGET,
                "positive_rate": float(df[TARGET].mean()),
                "compact_features": len(compact),
                "expanded_features": len(expanded),
            }
        ]
    )

    arrays_by_set = {
        "compact": split_arrays(splits, compact, TARGET),
    }
    feature_names_by_set = {"compact": compact}

    if len(expanded) > len(compact):
        arrays_by_set["expanded"] = split_arrays(splits, expanded, TARGET)
        feature_names_by_set["expanded"] = expanded

    add_corr_pruned_feature_set(
        arrays_by_set=arrays_by_set,
        feature_names_by_set=feature_names_by_set,
        splits=splits,
        base_features=compact,
        target=TARGET,
        threshold=0.95,
    )

    pca_summary = add_pca_feature_set(
        arrays_by_set=arrays_by_set,
        feature_names_by_set=feature_names_by_set,
        splits=splits,
        base_features=compact,
        target=TARGET,
        n_components=min(6, len(compact)),
    )

    configs = default_model_configs(random_state=42)
    metrics, results = run_classical_suite(
        arrays_by_feature_set=arrays_by_set,
        feature_names_by_set=feature_names_by_set,
        configs=configs,
    )

    corr_summary = correlation_summary(df, compact)

    dataset_summary.to_csv(TABLE_DIR / "dataset_summary.csv", index=False)
    split_summary.to_csv(TABLE_DIR / "split_summary.csv", index=False)
    metrics.to_csv(TABLE_DIR / "classical_metrics.csv", index=False)
    corr_summary.to_csv(TABLE_DIR / "feature_correlations.csv", index=False)
    pca_summary.to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)

    figure_paths = []
    figure_paths.append(save_correlation_heatmap(df, compact, FIGURE_DIR / "compact_feature_correlation.png"))
    figure_paths.append(save_metric_barplot(metrics, "val_pr_auc", FIGURE_DIR / "top_models_val_pr_auc.png"))
    figure_paths.append(save_metric_barplot(metrics, "test_pr_auc", FIGURE_DIR / "top_models_test_pr_auc.png"))
    figure_paths.append(save_pca_variance_plot(pca_summary, FIGURE_DIR / "pca_explained_variance.png"))

    top_result_items = {}
    for _, row in metrics.head(6).iterrows():
        key = (row["feature_set"], row["model_name"])
        if key in results:
            top_result_items[f"{key[0]}::{key[1]}"] = results[key]

    _, y_val = arrays_by_set[metrics.iloc[0]["feature_set"]]["val"]
    pr_path, roc_path = save_pr_roc_curves(
        top_result_items,
        y_val=y_val,
        out_pr_path=FIGURE_DIR / "validation_pr_curves.png",
        out_roc_path=FIGURE_DIR / "validation_roc_curves.png",
    )
    figure_paths.extend([pr_path, roc_path])

    best_key = (metrics.iloc[0]["feature_set"], metrics.iloc[0]["model_name"])
    best_result = results[best_key]
    X_val, y_val = arrays_by_set[best_key[0]]["val"]
    importance = permutation_importance_table(best_result, X_val, y_val)
    importance.to_csv(TABLE_DIR / "permutation_importance_best_model.csv", index=False)
    figure_paths.append(
        save_feature_importance_plot(
            importance,
            FIGURE_DIR / "permutation_importance_best_model.png",
            title=f"Permutation importance: {best_key[0]}::{best_key[1]}",
        )
    )

    notes = [
        "This report benchmarks fast classical tabular models before further ESN/QRC work.",
        "All preprocessing, PCA, correlation pruning, thresholds, and model fitting are fit from train/validation only according to their role.",
        "Primary model-selection metric shown here is validation PR-AUC because the stress class is imbalanced.",
        "Test metrics are reported for diagnosis, but model choice should be justified from validation metrics.",
    ]

    build_pdf_report(
        output_pdf=REPORT_DIR / "classical_baseline_report.pdf",
        title="Classical Baselines and Feature Diagnostics",
        notes=notes,
        tables={
            "Dataset summary": dataset_summary,
            "Split summary": split_summary,
            "Top classical baselines": metrics[[
                "feature_set",
                "model_name",
                "n_features",
                "val_pr_auc",
                "val_f1_class_1",
                "test_pr_auc",
                "test_f1_class_1",
            ]],
            "Highest feature correlations": corr_summary.head(15),
            "PCA explained variance": pca_summary,
            "Best-model permutation importance": importance.head(15),
        },
        figure_paths=figure_paths,
    )

    print(f"Wrote report to {REPORT_DIR / 'classical_baseline_report.pdf'}")
    print(f"Wrote tables to {TABLE_DIR}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
