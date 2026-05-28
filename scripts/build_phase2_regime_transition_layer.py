from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.loaders import load_phase2_volatility_data
from qpitome_qrc.data.pca import fit_transform_pca_splits_train_only
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.qrc.tfim_reservoir import (
    TFIMQRCConfig,
    _safe_feature_target_correlations,
    build_qrc_feature_matrix,
    make_qrc_sequence_splits,
)


def leaky_integrate_window(window: np.ndarray, leak: float = 0.3) -> np.ndarray:
    h = np.zeros(window.shape[1], dtype=float)
    out = np.zeros_like(window, dtype=float)
    for t, u_t in enumerate(window):
        h = (1.0 - leak) * h + leak * u_t
        out[t] = h
    return out


def leaky_integrate_windows(X: np.ndarray, leak: float = 0.3) -> np.ndarray:
    return np.stack([leaky_integrate_window(window, leak=leak) for window in X], axis=0)


def robust_topk_log_readout(
    H_train_raw: np.ndarray,
    H_val_raw: np.ndarray,
    H_test_raw: np.ndarray,
    y_train: np.ndarray,
    *,
    top_k: int = 240,
    alpha: float = 1000.0,
):
    lower = np.percentile(H_train_raw, 1.0, axis=0)
    upper = np.percentile(H_train_raw, 99.0, axis=0)

    H_train = np.clip(H_train_raw, lower, upper)
    H_val = np.clip(H_val_raw, lower, upper)
    H_test = np.clip(H_test_raw, lower, upper)

    corr = _safe_feature_target_correlations(H_train, y_train)
    k = min(top_k, H_train.shape[1])
    selected = np.argsort(np.abs(corr))[-k:]

    H_train = H_train[:, selected]
    H_val = H_val[:, selected]
    H_test = H_test[:, selected]

    scaler = StandardScaler()
    H_train_s = scaler.fit_transform(H_train)
    H_val_s = scaler.transform(H_val)
    H_test_s = scaler.transform(H_test)

    model = Ridge(alpha=alpha)
    model.fit(H_train_s, np.log(np.maximum(y_train, 1e-8)))

    return {
        "pred_train": np.exp(model.predict(H_train_s)),
        "pred_val": np.exp(model.predict(H_val_s)),
        "pred_test": np.exp(model.predict(H_test_s)),
        "selected": selected,
        "scaler": scaler,
        "model": model,
    }


def assign_regime(values: np.ndarray, calm_thr: float, turbulent_thr: float, crisis_thr: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    regimes = np.full(values.shape, "calm", dtype=object)
    regimes[values > calm_thr] = "elevated"
    regimes[values > turbulent_thr] = "turbulent"
    regimes[values > crisis_thr] = "crisis"
    return regimes


def transition_warning_score(forecast: np.ndarray, calm_thr: float, crisis_thr: float) -> np.ndarray:
    denom = max(crisis_thr - calm_thr, 1e-12)
    return np.clip((np.asarray(forecast, dtype=float) - calm_thr) / denom, 0.0, 1.0)


def assign_warning_level(forecast: np.ndarray, turbulent_thr: float, crisis_thr: float) -> np.ndarray:
    levels = np.full(np.asarray(forecast).shape, "none", dtype=object)
    levels[forecast > turbulent_thr] = "watch"
    levels[forecast > crisis_thr] = "alert"
    return levels


def make_timeline_plot(timeline: pd.DataFrame, out_path: Path, title: str, window: tuple[int, int] | None = None) -> None:
    plot_df = timeline.iloc[window[0] : window[1]].copy() if window is not None else timeline.copy()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(plot_df["date"], plot_df["realized_future_rv_20d"], label="Realized future RV 20d", linewidth=1.5)
    ax.plot(plot_df["date"], plot_df["qrc_forecast"], label="QRC forecast", linewidth=1.5)
    ax.axhline(plot_df["calm_threshold"].iloc[0], linestyle="--", linewidth=1.0, label="Calm/elevated threshold")
    ax.axhline(plot_df["turbulent_threshold"].iloc[0], linestyle="--", linewidth=1.0, label="Turbulent threshold")
    ax.axhline(plot_df["crisis_threshold"].iloc[0], linestyle="--", linewidth=1.0, label="Crisis threshold")

    watch = plot_df[plot_df["warning_level"] == "watch"]
    alert = plot_df[plot_df["warning_level"] == "alert"]
    if not watch.empty:
        ax.scatter(watch["date"], watch["qrc_forecast"], marker="^", s=35, label="Watch")
    if not alert.empty:
        ax.scatter(alert["date"], alert["qrc_forecast"], marker="^", s=55, label="Alert")

    ax.set_title(title)
    ax.set_ylabel("Realized / forecast volatility")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    out_table_dir = Path("results/tables")
    out_fig_dir = Path("results/figures")
    out_table_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    target = "future_rv_20d"
    df = load_phase2_volatility_data()
    splits = chronological_tabular_split(df)
    pca6 = fit_transform_pca_splits_train_only(
        splits,
        feature_columns=FEATURE_COLUMNS,
        target_columns=[target],
        n_components=6,
        prefix="pca6",
    )

    seq = make_qrc_sequence_splits(
        pca6.splits,
        feature_columns=pca6.feature_columns,
        target_column=target,
        lookback_days=40,
    )

    leaky_seq = {
        split: (leaky_integrate_windows(X, leak=0.3), y, dates)
        for split, (X, y, dates) in seq.items()
    }

    X_train, y_train, train_dates = leaky_seq["train"]
    X_val, y_val, val_dates = leaky_seq["val"]
    X_test, y_test, test_dates = leaky_seq["test"]

    config = TFIMQRCConfig(
        qubits=6,
        pca_components=6,
        lookback_days=40,
        anchor_count=10,
        anchor_policy="recent",
        observable_mode="zxzz",
        collect_anchor_features=True,
        topology="full",
        trotter_steps_per_anchor=3,
        virtual_nodes_per_anchor=3,
        coupling_scale=0.7,
        transverse_field=0.5,
        evolution_time=0.5,
        angle_max=np.pi / 2,
        ridge_alpha=1000.0,
        target_transform="log",
        seed=42,
        use_disorder=True,
        disorder_strength=0.20,
    )

    print("Building final QRC features...")
    H_train = build_qrc_feature_matrix(X_train, config, verbose=True)
    H_val = build_qrc_feature_matrix(X_val, config, verbose=False)
    H_test = build_qrc_feature_matrix(X_test, config, verbose=False)

    readout = robust_topk_log_readout(H_train, H_val, H_test, y_train, top_k=240, alpha=1000.0)

    calm_thr = float(np.quantile(y_train, 0.60))
    turbulent_thr = float(np.quantile(y_train, 0.80))
    crisis_thr = float(np.quantile(y_train, 0.90))

    timeline = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates).to_numpy(),
            "realized_future_rv_20d": y_test,
            "qrc_forecast": readout["pred_test"],
            "calm_threshold": calm_thr,
            "turbulent_threshold": turbulent_thr,
            "crisis_threshold": crisis_thr,
        }
    )
    timeline["realized_regime"] = assign_regime(
        timeline["realized_future_rv_20d"].to_numpy(), calm_thr, turbulent_thr, crisis_thr
    )
    timeline["forecast_regime"] = assign_regime(timeline["qrc_forecast"].to_numpy(), calm_thr, turbulent_thr, crisis_thr)
    timeline["transition_warning_score"] = transition_warning_score(timeline["qrc_forecast"].to_numpy(), calm_thr, crisis_thr)
    timeline["warning_level"] = assign_warning_level(timeline["qrc_forecast"].to_numpy(), turbulent_thr, crisis_thr)

    timeline_path = out_table_dir / "phase2_regime_transition_timeline.csv"
    timeline.to_csv(timeline_path, index=False)

    full_plot_path = out_fig_dir / "phase2_regime_transition_timeline.png"
    make_timeline_plot(timeline, full_plot_path, "Final QRC regime-transition interpretation layer")

    # Event-level case study: highest realized volatility event in the test split.
    event_idx = int(np.argmax(timeline["realized_future_rv_20d"].to_numpy()))
    start = max(event_idx - 80, 0)
    end = min(event_idx + 81, len(timeline))
    event_window = timeline.iloc[start:end].copy()
    event_window_path = out_table_dir / "phase2_regime_transition_case_study.csv"
    event_window.to_csv(event_window_path, index=False)

    case_plot_path = out_fig_dir / "phase2_regime_transition_case_study.png"
    make_timeline_plot(
        timeline,
        case_plot_path,
        "Event-level case study: highest realized-volatility test episode",
        window=(start, end),
    )

    event_row = timeline.iloc[event_idx]
    prior = timeline.iloc[max(event_idx - 20, 0) : event_idx + 1]
    first_watch = prior[prior["warning_level"].isin(["watch", "alert"])]

    if first_watch.empty:
        first_warning_date = "none within prior 20 observations"
    else:
        first_warning_date = str(first_watch.iloc[0]["date"].date())

    summary = pd.DataFrame(
        [
            {
                "event_date": str(event_row["date"].date()),
                "event_realized_future_rv_20d": float(event_row["realized_future_rv_20d"]),
                "event_qrc_forecast": float(event_row["qrc_forecast"]),
                "event_realized_regime": event_row["realized_regime"],
                "event_forecast_regime": event_row["forecast_regime"],
                "first_watch_or_alert_within_prior_20_obs": first_warning_date,
                "calm_threshold_q60_train": calm_thr,
                "turbulent_threshold_q80_train": turbulent_thr,
                "crisis_threshold_q90_train": crisis_thr,
            }
        ]
    )
    summary_path = out_table_dir / "phase2_regime_transition_case_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved:")
    print(f"  {timeline_path}")
    print(f"  {event_window_path}")
    print(f"  {summary_path}")
    print(f"  {full_plot_path}")
    print(f"  {case_plot_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
