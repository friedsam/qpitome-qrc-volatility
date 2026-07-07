from pathlib import Path

import pandas as pd

from qpitome_qrc.data.loaders import load_phase2_volatility_data
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast


OUT_DIR = Path("results/tables")
OUT_PATH = OUT_DIR / "phase2_persistence_baselines.csv"


BASELINE_SPECS = [
    {
        "model": "persistence_5d_to_5d",
        "target": "future_rv_5d",
        "prediction": "rv_5d",
    },
    {
        "model": "persistence_10d_to_5d",
        "target": "future_rv_5d",
        "prediction": "rv_10d",
    },
    {
        "model": "persistence_20d_to_20d",
        "target": "future_rv_20d",
        "prediction": "rv_20d",
    },
    {
        "model": "persistence_60d_to_20d",
        "target": "future_rv_20d",
        "prediction": "rv_60d",
    },
]


def evaluate_spec(split_name: str, split: pd.DataFrame, spec: dict) -> dict:
    y_true = split[spec["target"]].to_numpy(dtype=float)
    y_pred = split[spec["prediction"]].to_numpy(dtype=float)

    metrics = evaluate_volatility_forecast(y_true, y_pred)

    return {
        "split": split_name,
        "model": spec["model"],
        "target": spec["target"],
        "prediction": spec["prediction"],
        "n": len(split),
        "rmse": metrics.rmse,
        "qlike": metrics.qlike,
        "mz_alpha": metrics.mz_alpha,
        "mz_beta": metrics.mz_beta,
        "mz_r2": metrics.mz_r2,
    }


def main() -> None:
    df = load_phase2_volatility_data()
    splits = chronological_tabular_split(df)

    rows = []
    for split_name, split in splits.items():
        for spec in BASELINE_SPECS:
            rows.append(evaluate_spec(split_name, split, spec))

    results = pd.DataFrame(rows).sort_values(
        ["target", "split", "rmse"],
        ascending=[True, True, True],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
