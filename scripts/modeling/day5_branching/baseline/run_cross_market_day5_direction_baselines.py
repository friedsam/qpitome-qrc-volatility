from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

from qpitome_qrc.regimes.branch_intermediate_dynamics import IntermediateProbeConfig, build_first_passage_episode_table
from qpitome_qrc.regimes.branch_transition_path import add_transition_path_channels
from qpitome_qrc.regimes.market_portability import DEFAULT_MARKETS, audit_market_portability
from qpitome_qrc.regimes.crisis_clusters import build_resolved_episode_intervals

OUT = Path("results/modeling/day5_branching/baseline")
PREFIX = "cross_market_day5_direction_v1__"
FOLLOWUP = 120
LANDMARK = 5
MIN_TRAIN = 30
INPUTS = {
    "nikkei_225": Path("data/raw/portability/nikkei_225_fred_raw.csv"),
    "ftse_100": Path("data/raw/portability/ftse_100_raw.csv"),
    "russell_2000": Path("data/raw/portability/russell_2000_raw.csv"),
}


def complete(episodes, n_rows):
    return episodes[(n_rows - episodes["branch_idx"].astype(int) - 1) >= FOLLOWUP].copy()


def add_rows(market_key, market, daily, episodes, events):
    daily = add_transition_path_channels(daily)
    events = events[events["resolved_within_followup"] & events["event_type"].isin(["recovery", "relapse"])].copy()
    episodes = episodes.merge(events[["episode_id", "event_day", "event_type", "upper_barrier", "lower_barrier"]], on="episode_id")
    rows = []
    prices = daily["spy_adj_close"].to_numpy(float)
    for ep in episodes.itertuples(index=False):
        b = int(ep.branch_idx)
        if int(ep.event_day) <= LANDMARK or b + LANDMARK >= len(daily):
            continue
        branch_price = prices[b]
        p5 = prices[b + LANDMARK]
        r5 = p5 / branch_price - 1.0
        upper = float(ep.upper_barrier)
        lower = float(ep.lower_barrier)
        day = daily.iloc[b + LANDMARK]
        rows.append({
            "market_key": market_key,
            "market": market,
            "episode_id": int(ep.episode_id),
            "branch_date": ep.branch_date,
            "landmark_date": day["date"],
            "event_type": ep.event_type,
            "y_recovery": int(ep.event_type == "recovery"),
            "event_day": int(ep.event_day),
            "current_return_5d_from_branch": r5,
            "distance_to_recovery_barrier": upper - r5,
            "distance_to_relapse_barrier": r5 - lower,
            "barrier_width": upper - lower,
            "stress_ratio": float(ep.rv_20d / ep.branch_stress_cut),
            "drawdown_120d": float(ep.drawdown_120d),
            "rv_ratio_5_20_branch": float(ep.rv_ratio_5_20_branch),
            "return_5d_branch": float(ep.return_5d_branch),
            "rv_5d_change_5d_branch": float(ep.rv_5d_change_5d_branch),
            "worst_return_5d_in_prior_window": float(ep.worst_return_5d_in_prior_window),
            "downside_shock_pressure_day5": float(day["downside_shock_pressure"]),
        })
    return pd.DataFrame(rows)


def market_frame(key, path):
    daily, episodes, events, _ = audit_market_portability(path, key)
    episodes = complete(episodes, len(daily))
    events = events[events["episode_id"].isin(episodes["episode_id"])].copy()
    return add_rows(key, DEFAULT_MARKETS[key].market, daily, episodes, events), daily, episodes, events


def spy_frame():
    daily = pd.read_csv("data/processed/phase3_spy_vix_volatility_extended.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv("results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv", parse_dates=["branch_date"])
    episodes = complete(episodes, len(daily))
    events = build_first_passage_episode_table(daily, episodes, IntermediateProbeConfig(max_followup=FOLLOWUP))
    return add_rows("spy", "SPY", daily, episodes, events), daily, episodes, events


def fit_predict(train, test, cols, quadratic=False):
    if len(train) < MIN_TRAIN or train["y_recovery"].nunique() < 2:
        return np.nan
    steps = [("scale", StandardScaler())]
    if quadratic:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("logit", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")))
    model = Pipeline(steps)
    model.fit(train[cols].to_numpy(float), train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(test[cols].to_numpy(float))[0, 1])


def prequential_scores(frame):
    d1 = ["current_return_5d_from_branch", "distance_to_recovery_barrier", "distance_to_relapse_barrier", "barrier_width"]
    d2 = d1 + ["downside_shock_pressure_day5"]
    d3 = d2 + ["stress_ratio", "drawdown_120d", "rv_ratio_5_20_branch", "return_5d_branch", "rv_5d_change_5d_branch", "worst_return_5d_in_prior_window"]
    frame = frame.dropna(subset=d3 + ["y_recovery", "landmark_date"]).sort_values(["landmark_date", "market_key", "episode_id"]).reset_index(drop=True)
    preds = []
    for i, row in frame.iterrows():
        train = frame[frame["landmark_date"] < row["landmark_date"]]
        test = frame.iloc[[i]]
        if len(train) < MIN_TRAIN:
            continue
        p0 = float(np.clip(train["y_recovery"].mean(), 1e-6, 1 - 1e-6))
        preds.append({
            "row_id": i,
            "market_key": row["market_key"],
            "episode_id": row["episode_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
            "D0_prior": p0,
            "D1_geometry": fit_predict(train, test, d1),
            "D2_geometry_downside": fit_predict(train, test, d2),
            "D3_compact_quadratic": fit_predict(train, test, d3, quadratic=True),
        })
    return frame, pd.DataFrame(preds)


def metrics(preds):
    rows = []
    for model in ["D0_prior", "D1_geometry", "D2_geometry_downside", "D3_compact_quadratic"]:
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
    frames = []
    spy, _, _, _ = spy_frame(); frames.append(spy)
    for key, path in INPUTS.items():
        f, _, _, _ = market_frame(key, path); frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    frame, preds = prequential_scores(frame)
    m = metrics(preds)
    frame.to_csv(OUT / f"{PREFIX}day5_landmark_frame.csv", index=False)
    preds.to_csv(OUT / f"{PREFIX}calendar_prequential_predictions.csv", index=False)
    m.to_csv(OUT / f"{PREFIX}summary_metrics.csv", index=False)
    (OUT / f"{PREFIX}run_manifest.json").write_text(json.dumps({"landmark_day": LANDMARK, "min_train": MIN_TRAIN, "evaluation": "calendar-prequential, train landmark_date < test landmark_date"}, indent=2) + "\n")
    print("Cross-market day-5 direction baselines")
    print("Landmark rows:", len(frame))
    print("Predicted rows:", len(preds))
    print(m.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
