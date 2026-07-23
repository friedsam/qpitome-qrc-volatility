from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LATER_FOLDS = (4, 5, 6, 7, 8)
MODEL_ORDER = (
    "har",
    "crisis_template_only",
    "template_plus_qrc_unit",
    "template_plus_qrc_signed",
)
MODEL_LABELS = {
    "har": "HAR",
    "crisis_template_only": "Crisis template",
    "template_plus_qrc_unit": "Template + centered QRC (lambda=1)",
    "template_plus_qrc_signed": "Template + centered QRC (signed lambda)",
}


def _forecast_path(predictions: pd.DataFrame, *, label: int) -> pd.DataFrame:
    frame = predictions.loc[
        predictions["fold"].isin(LATER_FOLDS) & predictions["label"].eq(label)
    ]
    return (
        frame.groupby(["model_name", "horizon"], as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            har_pred=("har_pred", "mean"),
        )
    )


def _plot_forecast_path(
    predictions: pd.DataFrame,
    *,
    label: int,
    output_path: Path,
    title: str,
) -> None:
    paths = _forecast_path(predictions, label=label)
    har = paths.loc[paths["model_name"].eq("har")].sort_values("horizon")
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.plot(
        har["horizon"],
        har["y_true"],
        marker="o",
        linewidth=2.5,
        label="Realized",
    )
    for model_name in MODEL_ORDER:
        local = paths.loc[paths["model_name"].eq(model_name)].sort_values("horizon")
        axis.plot(
            local["horizon"],
            local["y_pred"],
            marker="o",
            linewidth=2.0,
            label=MODEL_LABELS[model_name],
        )
    axis.axvline(5, linestyle="--", linewidth=1.2)
    axis.set_title(title)
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Mean future log volatility")
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def render_graphs(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(run_dir / "predictions.csv.gz")
    comparison = pd.read_csv(run_dir / "increment_vs_template.csv")
    fit_parameters = pd.read_csv(run_dir / "selected_fit_parameters.csv")

    outputs: list[Path] = []
    for label, filename, title in (
        (1, "l5_transition_forecast_paths.png", "L5 transition paths: crisis template versus incremental QRC"),
        (0, "l5_control_forecast_paths.png", "Matched non-transition paths: false-positive cost"),
    ):
        output = plots_dir / filename
        _plot_forecast_path(
            predictions,
            label=label,
            output_path=output,
            title=title,
        )
        outputs.append(output)

    signed = predictions.loc[
        predictions["fold"].isin(LATER_FOLDS)
        & predictions["model_name"].eq("template_plus_qrc_signed")
    ].copy()
    signed["correction"] = signed["y_pred"] - signed["har_pred"]
    signed_path = (
        signed.groupby(["label", "horizon"], as_index=False)
        .agg(correction=("correction", "mean"))
    )
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for label, label_text in ((1, "L5 transitions"), (0, "Matched non-transitions")):
        local = signed_path.loc[signed_path["label"].eq(label)].sort_values("horizon")
        axis.plot(
            local["horizon"],
            local["correction"],
            marker="o",
            linewidth=2.2,
            label=label_text,
        )
    axis.axhline(0.0, linewidth=1.0)
    axis.axvline(5, linestyle="--", linewidth=1.2)
    axis.set_title("Signed centered-QRC correction: transition versus matched non-transition")
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Mean QRC correction to HAR")
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output = plots_dir / "signed_qrc_transition_vs_control_correction.png"
    figure.savefig(output, dpi=220)
    plt.close(figure)
    outputs.append(output)

    for metric, filename, title, ylabel in (
        (
            "delta_qlike_vs_template",
            "incremental_qlike_vs_template_by_seed.png",
            "Incremental QRC effect on transition QLIKE",
            "Delta QLIKE versus crisis template (negative is better)",
        ),
        (
            "delta_rmse_vs_template",
            "incremental_rmse_vs_template_by_seed.png",
            "Incremental QRC effect on transition RMSE",
            "Delta RMSE versus crisis template (negative is better)",
        ),
    ):
        figure, axis = plt.subplots(figsize=(9.5, 5.8))
        for model_name in ("template_plus_qrc_unit", "template_plus_qrc_signed"):
            local = comparison.loc[comparison["model_name"].eq(model_name)].sort_values(
                "selection_seed"
            )
            axis.plot(
                local["selection_seed"].astype(str),
                local[metric],
                marker="o",
                linewidth=2.0,
                label=MODEL_LABELS[model_name],
            )
        axis.axhline(0.0, linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Selection seed")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        output = plots_dir / filename
        figure.savefig(output, dpi=220)
        plt.close(figure)
        outputs.append(output)

    fit_parameters = fit_parameters.copy()
    fit_parameters["cell"] = (
        fit_parameters["selection_seed"].astype(str).str[-2:]
        + "/F"
        + fit_parameters["fold"].astype(str)
    )
    figure, axis = plt.subplots(figsize=(12.0, 5.8))
    x = np.arange(len(fit_parameters))
    axis.plot(x, fit_parameters["lambda_early"], marker="o", linewidth=1.5, label="Early lambda")
    axis.plot(x, fit_parameters["lambda_late"], marker="o", linewidth=1.5, label="Late lambda")
    axis.axhline(0.0, linewidth=1.0)
    axis.set_title("Selected signed lambdas across seed/fold fits")
    axis.set_xlabel("Seed suffix / fold")
    axis.set_ylabel("Selected lambda")
    axis.set_xticks(x)
    axis.set_xticklabels(fit_parameters["cell"], rotation=90)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output = plots_dir / "selected_signed_lambdas.png"
    figure.savefig(output, dpi=220)
    plt.close(figure)
    outputs.append(output)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render diagnostic graphs for an existing L5 template-increment run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render_graphs(args.run_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
