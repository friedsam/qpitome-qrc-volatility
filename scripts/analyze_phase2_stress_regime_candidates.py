from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.loaders import load_phase2_volatility_data
from qpitome_qrc.data.splits import chronological_tabular_split


def percentile_from_train(train_values: pd.Series, values: pd.Series) -> pd.Series:
    train = train_values.dropna().to_numpy(dtype=float)
    x = values.to_numpy(dtype=float)
    return pd.Series(np.searchsorted(np.sort(train), x, side="right") / max(len(train), 1), index=values.index)


def flag_metrics(signal_flag: pd.Series, event_flag: pd.Series) -> dict[str, float]:
    signal = signal_flag.astype(bool).to_numpy()
    event = event_flag.astype(bool).to_numpy()
    tp = int(np.sum(signal & event))
    fp = int(np.sum(signal & ~event))
    fn = int(np.sum(~signal & event))
    tn = int(np.sum(~signal & ~event))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "event_rate": float(event.mean()),
        "signal_rate": float(signal.mean()),
    }


def make_stress_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Directional and stress-oriented transformations.
    out["spy_selloff_stress"] = -out["spy_log_return"]
    out["spy_oc_selloff_stress"] = -out["spy_log_oc_return"]
    out["drawdown_stress"] = -out["spy_drawdown_20d"] if out["spy_drawdown_20d"].median() < 0 else out["spy_drawdown_20d"]
    out["volume_stress"] = out["spy_log_volume_change"].abs()
    out["vix_level_stress"] = out["vix_close"]
    out["vix_change_stress"] = out["vix_log_change"].clip(lower=0.0)
    out["vix_range_stress"] = out["vix_log_hl_range"]
    out["realized_vol_stress"] = out["rv_20d"]
    out["short_vol_acceleration"] = out["rv_slope_5_20"]
    out["medium_vol_acceleration"] = out["rv_slope_20_60"]
    return out


def main() -> None:
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_phase2_volatility_data()
    splits = chronological_tabular_split(df)
    splits = {name: make_stress_features(split.reset_index(drop=True)) for name, split in splits.items()}

    train = splits["train"]
    test = splits["test"]

    target = "future_rv_20d"

    candidate_features = [
        "rv_5d",
        "rv_10d",
        "rv_20d",
        "rv_60d",
        "rv_ratio_5_20",
        "rv_ratio_20_60",
        "rv_slope_5_20",
        "rv_slope_20_60",
        "spy_abs_log_return",
        "spy_squared_log_return",
        "spy_log_hl_range",
        "spy_selloff_stress",
        "spy_oc_selloff_stress",
        "drawdown_stress",
        "spy_log_volume",
        "spy_log_volume_change",
        "volume_stress",
        "vix_close",
        "vix_log_change",
        "vix_abs_log_change",
        "vix_log_hl_range",
        "vix_ma_5d",
        "vix_std_5d",
        "vix_ma_20d",
        "vix_std_20d",
        "vix_level_stress",
        "vix_change_stress",
        "vix_range_stress",
        "realized_vol_stress",
        "short_vol_acceleration",
        "medium_vol_acceleration",
    ]
    candidate_features = [c for c in candidate_features if c in train.columns]

    # Realized future stress events. These are not hand labels; they are train-calibrated outcome events.
    high_vol_thr = float(train[target].quantile(0.80))
    extreme_vol_thr = float(train[target].quantile(0.90))
    crisis_vol_thr = float(train[target].quantile(0.95))

    for split_name, split in splits.items():
        split["future_high_vol_event_q80"] = split[target] > high_vol_thr
        split["future_extreme_vol_event_q90"] = split[target] > extreme_vol_thr
        split["future_crisis_candidate_q95"] = split[target] > crisis_vol_thr

    test = splits["test"]

    # Feature ranking: which contemporaneous indicators are most associated with future high/extreme volatility?
    ranking_rows = []
    for feature in candidate_features:
        train_feature = train[feature]
        test_feature = test[feature]
        q80 = float(train_feature.quantile(0.80))
        q90 = float(train_feature.quantile(0.90))
        q95 = float(train_feature.quantile(0.95))

        # For each feature, q80 flag is an early stress candidate.
        flag_q80 = test_feature > q80
        flag_q90 = test_feature > q90

        for event_name in ["future_high_vol_event_q80", "future_extreme_vol_event_q90", "future_crisis_candidate_q95"]:
            metrics_q80 = flag_metrics(flag_q80, test[event_name])
            metrics_q90 = flag_metrics(flag_q90, test[event_name])
            corr = float(np.corrcoef(test_feature, test[target])[0, 1]) if test_feature.std() > 0 else np.nan
            crisis_mean = float(test.loc[test[event_name], feature].mean()) if test[event_name].any() else np.nan
            non_mean = float(test.loc[~test[event_name], feature].mean()) if (~test[event_name]).any() else np.nan
            ranking_rows.append(
                {
                    "feature": feature,
                    "event": event_name,
                    "feature_q80_train": q80,
                    "feature_q90_train": q90,
                    "feature_q95_train": q95,
                    "test_corr_with_future_rv_20d": corr,
                    "event_mean": crisis_mean,
                    "non_event_mean": non_mean,
                    "event_minus_non_mean": crisis_mean - non_mean,
                    "q80_precision": metrics_q80["precision"],
                    "q80_recall": metrics_q80["recall"],
                    "q80_f1": metrics_q80["f1"],
                    "q80_signal_rate": metrics_q80["signal_rate"],
                    "q90_precision": metrics_q90["precision"],
                    "q90_recall": metrics_q90["recall"],
                    "q90_f1": metrics_q90["f1"],
                    "q90_signal_rate": metrics_q90["signal_rate"],
                }
            )

    ranking = pd.DataFrame(ranking_rows).sort_values(["event", "q80_f1"], ascending=[True, False])
    ranking.to_csv(out_dir / "phase2_stress_feature_ranking.csv", index=False)

    # Transparent candidate composite stress rules. These are diagnostic, not optimized classifiers.
    def add_percentile_flags(split: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
        out = split.copy()
        q = lambda col, p: float(train_ref[col].quantile(p))

        out["flag_forecastless_realized_vol"] = out["rv_20d"] > q("rv_20d", 0.80)
        out["flag_vix_level"] = out["vix_close"] > q("vix_close", 0.80)
        out["flag_vix_range"] = out["vix_log_hl_range"] > q("vix_log_hl_range", 0.80)
        out["flag_drawdown"] = out["drawdown_stress"] > q("drawdown_stress", 0.80)
        out["flag_selloff"] = out["spy_selloff_stress"] > q("spy_selloff_stress", 0.80)
        out["flag_volume"] = out["volume_stress"] > q("volume_stress", 0.80)
        out["flag_vol_acceleration"] = out["rv_slope_5_20"] > q("rv_slope_5_20", 0.80)

        flag_cols = [
            "flag_forecastless_realized_vol",
            "flag_vix_level",
            "flag_vix_range",
            "flag_drawdown",
            "flag_selloff",
            "flag_volume",
            "flag_vol_acceleration",
        ]
        out["contemporaneous_stress_score"] = out[flag_cols].sum(axis=1)
        out["stress_ladder_candidate"] = pd.cut(
            out["contemporaneous_stress_score"],
            bins=[-0.1, 0.5, 1.5, 2.5, len(flag_cols) + 0.5],
            labels=["normal", "watch", "warning", "crisis_like"],
        ).astype(str)

        out["direction_tag"] = "neutral"
        out.loc[out["spy_log_return"] > q("spy_log_return", 0.70), "direction_tag"] = "rally"
        out.loc[out["spy_log_return"] < q("spy_log_return", 0.30), "direction_tag"] = "selloff"
        return out

    profiled = add_percentile_flags(test, train)
    profiled.to_csv(out_dir / "phase2_stress_ladder_candidate_timeline.csv", index=False)

    ladder_rows = []
    for score_thr in [1, 2, 3, 4]:
        signal = profiled["contemporaneous_stress_score"] >= score_thr
        for event_name in ["future_high_vol_event_q80", "future_extreme_vol_event_q90", "future_crisis_candidate_q95"]:
            row = {"signal": f"stress_score_ge_{score_thr}", "event": event_name, "score_threshold": score_thr}
            row.update(flag_metrics(signal, profiled[event_name]))
            ladder_rows.append(row)

    ladder_eval = pd.DataFrame(ladder_rows).sort_values(["event", "f1"], ascending=[True, False])
    ladder_eval.to_csv(out_dir / "phase2_stress_ladder_candidate_evaluation.csv", index=False)

    # Profile the most severe stress-score days and future high-volatility days.
    top_stress = profiled.sort_values("contemporaneous_stress_score", ascending=False).head(40)
    top_stress.to_csv(out_dir / "phase2_top_stress_score_days.csv", index=False)

    top_future_vol = profiled.sort_values(target, ascending=False).head(40)
    top_future_vol.to_csv(out_dir / "phase2_top_future_volatility_days.csv", index=False)

    print("Saved stress-regime candidate analysis tables:")
    print("  results/tables/phase2_stress_feature_ranking.csv")
    print("  results/tables/phase2_stress_ladder_candidate_timeline.csv")
    print("  results/tables/phase2_stress_ladder_candidate_evaluation.csv")
    print("  results/tables/phase2_top_stress_score_days.csv")
    print("  results/tables/phase2_top_future_volatility_days.csv")
    print("\nTop feature candidates for future_extreme_vol_event_q90 by q80 F1:")
    print(
        ranking[ranking["event"] == "future_extreme_vol_event_q90"]
        .head(12)[["feature", "q80_precision", "q80_recall", "q80_f1", "test_corr_with_future_rv_20d"]]
        .to_string(index=False)
    )
    print("\nCandidate ladder evaluation:")
    print(ladder_eval.to_string(index=False))


if __name__ == "__main__":
    main()
