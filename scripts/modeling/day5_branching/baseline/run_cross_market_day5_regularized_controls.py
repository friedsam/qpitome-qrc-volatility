from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5.protocol import D1, MIN_TRAIN
from qpitome_qrc.evaluation.binary import (
    fit_transformed_predict,
    transformed_logistic_pipeline,
)
from qpitome_qrc.evaluation.scoring import binary_summary

BASE = Path("results/modeling/day5_branching/baseline/cross_market_day5_direction_v1")
OUT = Path("results/modeling/day5_branching/baseline/cross_market_day5_regularized_controls_v1")

G = D1
S = [
    "stress_ratio",
    "drawdown_120d",
    "rv_ratio_5_20_branch",
    "return_5d_branch",
    "rv_5d_change_5d_branch",
    "worst_return_5d_in_prior_window",
]
D = ["downside_shock_pressure_day5"]
MODELS = {
    "D1_geometry_linear": {"cols": G, "kind": "linear", "c": 1.0},
    "D2_geometry_downside_linear": {"cols": G + D, "kind": "linear", "c": 1.0},
    "L1_geometry_state_linear_c01": {"cols": G + S, "kind": "linear", "c": 0.1},
    "L2_geometry_quad_c005": {"cols": G, "kind": "poly2", "c": 0.05},
    "L3_geometry_downside_quad_c005": {"cols": G + D, "kind": "poly2", "c": 0.05},
    "L4_state_rbf32_c005": {"cols": G + D + S, "kind": "rbf32", "c": 0.05},
}

# Backward-compatible historical names.
make_model = transformed_logistic_pipeline


def predict_one(train, test, spec):
    if len(train) < MIN_TRAIN or train["y_recovery"].nunique() < 2:
        return np.nan
    return fit_transformed_predict(
        train[spec["cols"]].to_numpy(float),
        train["y_recovery"].to_numpy(int),
        test[spec["cols"]].to_numpy(float),
        kind=spec["kind"],
        C=spec["c"],
    )


def metrics(preds):
    return pd.DataFrame([binary_summary(preds, model) for model in MODELS])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(BASE / "day5_landmark_frame.csv", parse_dates=["landmark_date"])
    all_cols = sorted(set(sum([spec["cols"] for spec in MODELS.values()], [])))
    frame = frame.dropna(
        subset=all_cols + ["y_recovery", "landmark_date"]
    ).sort_values(
        ["landmark_date", "market_key", "episode_id"]
    ).reset_index(drop=True)
    rows = []
    for index, row in frame.iterrows():
        train = frame[frame["landmark_date"] < row["landmark_date"]]
        if len(train) < MIN_TRAIN:
            continue
        test = frame.iloc[[index]]
        out = {
            "row_id": index,
            "market_key": row["market_key"],
            "episode_id": row["episode_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
        }
        for name, spec in MODELS.items():
            out[name] = predict_one(train, test, spec)
        rows.append(out)
    preds = pd.DataFrame(rows)
    summary = metrics(preds)
    preds.to_csv(OUT / "calendar_prequential_predictions.csv", index=False)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    print("Cross-market day-5 regularized controls")
    print("Rows:", len(frame), "Predictions:", len(preds))
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
