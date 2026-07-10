from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE = Path("results/baselines/cross_market_day5_direction_v1")
OUT = Path("results/baselines/cross_market_day5_regularized_controls_v1")
MIN_TRAIN = 30

G = ["current_return_5d_from_branch", "distance_to_recovery_barrier", "distance_to_relapse_barrier", "barrier_width"]
S = ["stress_ratio", "drawdown_120d", "rv_ratio_5_20_branch", "return_5d_branch", "rv_5d_change_5d_branch", "worst_return_5d_in_prior_window"]
D = ["downside_shock_pressure_day5"]
MODELS = {
    "D1_geometry_linear": {"cols": G, "kind": "linear", "c": 1.0},
    "D2_geometry_downside_linear": {"cols": G + D, "kind": "linear", "c": 1.0},
    "L1_geometry_state_linear_c01": {"cols": G + S, "kind": "linear", "c": 0.1},
    "L2_geometry_quad_c005": {"cols": G, "kind": "poly2", "c": 0.05},
    "L3_geometry_downside_quad_c005": {"cols": G + D, "kind": "poly2", "c": 0.05},
    "L4_state_rbf32_c005": {"cols": G + D + S, "kind": "rbf32", "c": 0.05},
}


def make_model(kind, c):
    steps = [("scale", StandardScaler())]
    if kind == "poly2":
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    elif kind == "rbf32":
        steps.append(("rbf", RBFSampler(gamma=0.25, n_components=32, random_state=17)))
    elif kind != "linear":
        raise ValueError(kind)
    steps.append(("logit", LogisticRegression(C=c, max_iter=5000, solver="lbfgs")))
    return Pipeline(steps)


def predict_one(train, test, spec):
    if len(train) < MIN_TRAIN or train["y_recovery"].nunique() < 2:
        return np.nan
    model = make_model(spec["kind"], spec["c"])
    model.fit(train[spec["cols"]].to_numpy(float), train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(test[spec["cols"]].to_numpy(float))[0, 1])


def metrics(preds):
    rows = []
    for model in MODELS:
        use = preds[["y", model]].dropna()
        y = use["y"].to_numpy(int)
        p = np.clip(use[model].to_numpy(float), 1e-6, 1 - 1e-6)
        rows.append({
            "model": model,
            "n": len(use),
            "recovery_rate": float(y.mean()) if len(y) else np.nan,
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "logloss": float(log_loss(y, p)) if len(y) else np.nan,
            "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(BASE / "day5_landmark_frame.csv", parse_dates=["landmark_date"])
    all_cols = sorted(set(sum([spec["cols"] for spec in MODELS.values()], [])))
    frame = frame.dropna(subset=all_cols + ["y_recovery", "landmark_date"]).sort_values(["landmark_date", "market_key", "episode_id"]).reset_index(drop=True)
    rows = []
    for i, row in frame.iterrows():
        train = frame[frame["landmark_date"] < row["landmark_date"]]
        if len(train) < MIN_TRAIN:
            continue
        test = frame.iloc[[i]]
        out = {"row_id": i, "market_key": row["market_key"], "episode_id": row["episode_id"], "landmark_date": row["landmark_date"], "y": int(row["y_recovery"])}
        for name, spec in MODELS.items():
            out[name] = predict_one(train, test, spec)
        rows.append(out)
    preds = pd.DataFrame(rows)
    m = metrics(preds)
    preds.to_csv(OUT / "calendar_prequential_predictions.csv", index=False)
    m.to_csv(OUT / "summary_metrics.csv", index=False)
    print("Cross-market day-5 regularized controls")
    print("Rows:", len(frame), "Predictions:", len(preds))
    print(m.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
