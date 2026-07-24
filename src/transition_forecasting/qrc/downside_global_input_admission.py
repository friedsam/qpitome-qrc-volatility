from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from transition_forecasting.qrc.input_admission_assay import (
    HAR_COLUMNS,
    PREQ_COLUMNS,
    TARGET_COLUMNS,
    chronological_inner_split,
    load_evaluation_manifest,
    load_ohlc_panel,
    qlike_loss,
    rmse_loss,
)

LOCAL_CHANNELS = (
    "local_downside_z",
    "local_drawdown_60",
    "local_downside_share_5",
)
GLOBAL_CHANNELS = (
    "global_fraction_negative",
    "global_fraction_severe",
    "global_median_z_return",
    "global_z_dispersion",
)
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "local_downside_state": LOCAL_CHANNELS,
    "lagged_global_breadth": GLOBAL_CHANNELS,
    "local_plus_global": LOCAL_CHANNELS + GLOBAL_CHANNELS,
}
REPRESENTATIONS = ("summary", "sequence")


@dataclass(frozen=True)
class DownsideGlobalAdmissionConfig:
    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    lead: int = 5
    window: int = 40
    local_volatility_lookback: int = 20
    drawdown_lookback: int = 60
    semivariance_window: int = 5
    minimum_peer_markets: int = 8
    inner_holdout_fraction: float = 0.25
    path_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    gate_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if self.lead < 1 or self.window < 10:
            raise ValueError("lead and window must be positive")
        if self.local_volatility_lookback < 5:
            raise ValueError("local_volatility_lookback must be at least five")
        if self.drawdown_lookback < self.window:
            raise ValueError("drawdown_lookback must be at least the input window")
        if self.semivariance_window < 2:
            raise ValueError("semivariance_window must be at least two")
        if self.minimum_peer_markets < 2:
            raise ValueError("minimum_peer_markets must be at least two")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not self.path_alphas or any(value <= 0 for value in self.path_alphas):
            raise ValueError("path_alphas must be positive")
        if not self.gate_cs or any(value <= 0 for value in self.gate_cs):
            raise ValueError("gate_cs must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rolling_downside_share(returns: pd.Series, window: int) -> pd.Series:
    negative = returns.clip(upper=0.0).pow(2)
    positive = returns.clip(lower=0.0).pow(2)
    minimum = max(2, window // 2)
    downside = negative.rolling(window, min_periods=minimum).sum()
    total = (negative + positive).rolling(window, min_periods=minimum).sum()
    return downside / total.where(total > 1e-16)


def build_local_state(
    panel: dict[str, pd.DataFrame],
    config: DownsideGlobalAdmissionConfig,
) -> dict[str, pd.DataFrame]:
    """Construct causal, scale-free downside state for every market."""

    output: dict[str, pd.DataFrame] = {}
    for ticker, values in panel.items():
        close = pd.to_numeric(values["close"], errors="coerce").where(lambda x: x > 0)
        log_close = np.log(close)
        returns = log_close.diff()
        prior_scale = (
            returns.shift(1)
            .rolling(
                config.local_volatility_lookback,
                min_periods=max(5, config.local_volatility_lookback // 2),
            )
            .std()
        )
        standardized = returns / prior_scale.where(prior_scale > 1e-12)
        rolling_peak = log_close.rolling(
            config.drawdown_lookback,
            min_periods=max(10, config.drawdown_lookback // 3),
        ).max()
        output[str(ticker)] = pd.DataFrame(
            {
                "local_downside_z": standardized.clip(upper=0.0),
                "local_drawdown_60": log_close - rolling_peak,
                "local_downside_share_5": _rolling_downside_share(
                    returns, config.semivariance_window
                ),
            },
            index=values.index,
        ).replace([np.inf, -np.inf], np.nan)
    return output


def _standardized_return_panel(
    panel: dict[str, pd.DataFrame],
    config: DownsideGlobalAdmissionConfig,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for ticker, values in panel.items():
        if str(ticker) == "^VIX":
            continue
        close = pd.to_numeric(values["close"], errors="coerce").where(lambda x: x > 0)
        returns = np.log(close).diff()
        prior_scale = (
            returns.shift(1)
            .rolling(
                config.local_volatility_lookback,
                min_periods=max(5, config.local_volatility_lookback // 2),
            )
            .std()
        )
        columns[str(ticker)] = returns / prior_scale.where(prior_scale > 1e-12)
    return pd.DataFrame(columns).sort_index().replace([np.inf, -np.inf], np.nan)


def build_leave_one_out_global_breadth(
    panel: dict[str, pd.DataFrame],
    config: DownsideGlobalAdmissionConfig,
) -> dict[str, pd.DataFrame]:
    """Build global breadth from peers only, lagged by a full union-market date."""

    standardized = _standardized_return_panel(panel, config)
    output: dict[str, pd.DataFrame] = {}
    for target_ticker in panel:
        peers = standardized.drop(columns=[str(target_ticker)], errors="ignore")
        count = peers.notna().sum(axis=1)
        median = peers.median(axis=1, skipna=True)
        dispersion = peers.sub(median, axis=0).abs().median(axis=1, skipna=True)
        contemporaneous = pd.DataFrame(
            {
                "global_fraction_negative": peers.lt(0.0).sum(axis=1)
                / count.replace(0, np.nan),
                "global_fraction_severe": peers.lt(-1.0).sum(axis=1)
                / count.replace(0, np.nan),
                "global_median_z_return": median,
                "global_z_dispersion": dispersion,
                "global_peer_count": count.astype(float),
            },
            index=standardized.index,
        )
        # Daily files have no reliable exchange-close timestamps. Shift by one full
        # union-market date so a target origin can never use another market's value
        # carrying the same calendar date.
        output[str(target_ticker)] = contemporaneous.shift(1)
    return output


def extract_state_window(
    local_state: dict[str, pd.DataFrame],
    global_state: dict[str, pd.DataFrame],
    ticker: str,
    origin_date: pd.Timestamp,
    config: DownsideGlobalAdmissionConfig,
) -> dict[str, np.ndarray] | None:
    if ticker not in local_state or ticker not in global_state:
        return None
    local = local_state[ticker].loc[:origin_date].dropna().tail(config.window)
    if len(local) != config.window:
        return None
    global_window = global_state[ticker].reindex(local.index)
    required = list(GLOBAL_CHANNELS) + ["global_peer_count"]
    if global_window[required].isna().any().any():
        return None
    if (global_window["global_peer_count"] < config.minimum_peer_markets).any():
        return None
    result = {channel: local[channel].to_numpy(dtype=float) for channel in LOCAL_CHANNELS}
    result.update(
        {
            channel: global_window[channel].to_numpy(dtype=float)
            for channel in GLOBAL_CHANNELS
        }
    )
    if any(values.shape != (config.window,) for values in result.values()):
        raise RuntimeError("state-window extraction produced an invalid shape")
    if any(not np.isfinite(values).all() for values in result.values()):
        return None
    return result


def _summary(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    payload = [float(data[-1])]
    for width in (5, 20, len(data)):
        local = data[-min(width, len(data)) :]
        payload.extend(
            [float(local.mean()), float(local.std()), float(local.min()), float(local.max())]
        )
    return np.asarray(payload, dtype=float)


def build_feature_tables(
    manifest: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: DownsideGlobalAdmissionConfig,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame, dict[str, np.ndarray]]:
    config.validate()
    local_state = build_local_state(panel, config)
    global_state = build_leave_one_out_global_breadth(panel, config)
    channels = {
        name: np.full((len(manifest), config.window), np.nan, dtype=float)
        for name in LOCAL_CHANNELS + GLOBAL_CHANNELS
    }
    valid = np.zeros(len(manifest), dtype=bool)
    for row_index, row in manifest.iterrows():
        extracted = extract_state_window(
            local_state,
            global_state,
            str(row["ticker"]),
            pd.Timestamp(row["origin_date"]),
            config,
        )
        if extracted is None:
            continue
        for name, values in extracted.items():
            channels[name][row_index] = values
        valid[row_index] = True

    matrices: dict[tuple[str, str], np.ndarray] = {}
    for feature_set, names in FEATURE_SETS.items():
        matrices[(feature_set, "sequence")] = np.concatenate(
            [channels[name] for name in names], axis=1
        )
        matrices[(feature_set, "summary")] = np.vstack(
            [
                np.concatenate([_summary(channels[name][row]) for name in names])
                for row in range(len(manifest))
            ]
        )
    availability = manifest[
        ["fold", "sample_id", "lead", "label", "ticker", "origin_date"]
    ].copy()
    availability["downside_global_valid"] = valid
    return matrices, availability, channels


def causal_har_path(frame: pd.DataFrame) -> np.ndarray:
    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = frame[list(HAR_COLUMNS)].to_numpy(dtype=float)
    residual = frame[list(PREQ_COLUMNS)].to_numpy(dtype=float)
    valid = frame["prequential_valid"].to_numpy(dtype=bool)
    causal = har.copy()
    causal[valid] = y[valid] - residual[valid]
    return causal


def fit_select_occurrence(
    features: np.ndarray,
    labels: np.ndarray,
    dates: pd.Series,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: DownsideGlobalAdmissionConfig,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    x = np.asarray(features, dtype=float)
    finite = np.isfinite(x).all(axis=1)
    target = np.asarray(labels, dtype=int)
    train = np.asarray(train_mask, dtype=bool) & finite
    validation = np.asarray(validation_mask, dtype=bool) & finite
    inner_fit, inner_tune = chronological_inner_split(
        dates, train, config.inner_holdout_fraction
    )
    for name, mask in (("inner_fit", inner_fit), ("inner_tune", inner_tune), ("train", train)):
        if set(np.unique(target[mask])) != {0, 1}:
            raise ValueError(f"{name} must contain both L5 classes")

    rows: list[dict[str, float]] = []
    for c_value in config.gate_cs:
        scaler = StandardScaler().fit(x[inner_fit])
        fit_design = scaler.transform(x[inner_fit])
        tune_design = scaler.transform(x[inner_tune])
        model = LogisticRegression(
            C=float(c_value),
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ).fit(fit_design, target[inner_fit])
        score = model.predict_proba(tune_design)[:, 1]
        rows.append(
            {
                "c_value": float(c_value),
                "inner_average_precision": float(
                    average_precision_score(target[inner_tune], score)
                ),
                "inner_roc_auc": float(roc_auc_score(target[inner_tune], score)),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            -row["inner_average_precision"],
            -row["inner_roc_auc"],
            row["c_value"],
        ),
    )
    scaler = StandardScaler().fit(x[train])
    model = LogisticRegression(
        C=float(selected["c_value"]),
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    ).fit(scaler.transform(x[train]), target[train])
    probability = np.full(len(x), np.nan, dtype=float)
    probability[validation] = model.predict_proba(
        scaler.transform(x[validation])
    )[:, 1]
    return probability, dict(selected), pd.DataFrame(rows)


def fit_select_direct_path(
    features: np.ndarray,
    har_path: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    dates: pd.Series,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: DownsideGlobalAdmissionConfig,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    raw = np.column_stack(
        [np.asarray(har_path, dtype=float), np.asarray(features, dtype=float)]
    )
    finite = np.isfinite(raw).all(axis=1)
    y = np.asarray(targets, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    fit = np.asarray(fit_mask, dtype=bool) & finite
    validation = np.asarray(validation_mask, dtype=bool) & finite
    inner_fit, inner_tune = chronological_inner_split(
        dates, fit, config.inner_holdout_fraction
    )
    inner_transition = inner_tune & labels_array.astype(bool)
    if not inner_transition.any():
        raise ValueError("inner path holdout contains no transitions")

    rows: list[dict[str, float]] = []
    for alpha in config.path_alphas:
        scaler = StandardScaler().fit(raw[inner_fit])
        model = Ridge(alpha=float(alpha)).fit(
            scaler.transform(raw[inner_fit]), y[inner_fit]
        )
        tune_prediction = model.predict(scaler.transform(raw[inner_tune]))
        transition_position = labels_array[inner_tune].astype(bool)
        rows.append(
            {
                "alpha": float(alpha),
                "inner_transition_qlike": qlike_loss(
                    y[inner_tune][transition_position],
                    tune_prediction[transition_position],
                ),
                "inner_overall_qlike": qlike_loss(y[inner_tune], tune_prediction),
                "inner_overall_rmse": rmse_loss(y[inner_tune], tune_prediction),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["inner_transition_qlike"],
            row["inner_overall_rmse"],
            row["alpha"],
        ),
    )
    scaler = StandardScaler().fit(raw[fit])
    model = Ridge(alpha=float(selected["alpha"])).fit(
        scaler.transform(raw[fit]), y[fit]
    )
    prediction = np.full_like(y, np.nan, dtype=float)
    prediction[validation] = model.predict(scaler.transform(raw[validation]))
    return prediction, dict(selected), pd.DataFrame(rows)


def _path_metrics(
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    prediction: np.ndarray,
    validation: np.ndarray,
    *,
    fold: int,
    feature_set: str,
    representation: str,
) -> list[dict[str, object]]:
    scopes = {
        "all": validation,
        "control": validation & frame["label"].eq(0).to_numpy(),
        "transition": validation & frame["label"].eq(1).to_numpy(),
    }
    common = np.isfinite(prediction).all(axis=1) & np.isfinite(har).all(axis=1)
    rows: list[dict[str, object]] = []
    for model_name, forecast in (("har", har), ("har_input_direct_path", prediction)):
        for scope, mask in scopes.items():
            use = np.asarray(mask, dtype=bool) & common
            if not use.any():
                continue
            rows.append(
                {
                    "fold": int(fold),
                    "feature_set": feature_set,
                    "representation": representation,
                    "model": model_name,
                    "scope": scope,
                    "samples": int(use.sum()),
                    "qlike": qlike_loss(y[use], forecast[use]),
                    "rmse": rmse_loss(y[use], forecast[use]),
                    "mean_correction_vs_har": float(np.mean(forecast[use] - har[use])),
                }
            )
    return rows


def _plot_inputs(
    channels: dict[str, np.ndarray],
    frame: pd.DataFrame,
    validation: np.ndarray,
    names: tuple[str, ...],
    title: str,
    path: Path,
) -> None:
    local_frame = frame.loc[validation].reset_index(drop=True)
    figure, axes = plt.subplots(len(names), 1, figsize=(9.0, 2.7 * len(names)), sharex=True)
    for axis, name in zip(np.atleast_1d(axes), names, strict=True):
        values = channels[name][validation]
        for label, label_name in ((0, "controls"), (1, "transitions")):
            mask = local_frame["label"].eq(label).to_numpy()
            if mask.any():
                axis.plot(
                    np.arange(-values.shape[1] + 1, 1),
                    np.nanmean(values[mask], axis=0),
                    label=label_name,
                )
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_title(title)
    np.atleast_1d(axes)[0].legend()
    np.atleast_1d(axes)[-1].set_xlabel("Trading days before origin")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _render_plots(
    occurrence: pd.DataFrame,
    path_metrics: pd.DataFrame,
    channels: dict[str, np.ndarray],
    frame: pd.DataFrame,
    validation: np.ndarray,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    occurrence_mean = occurrence.groupby(
        ["feature_set", "representation"], as_index=False
    )["average_precision"].mean()
    labels = occurrence_mean["feature_set"] + " / " + occurrence_mean["representation"]
    figure, axis = plt.subplots(figsize=(9.0, 5.3))
    axis.barh(np.arange(len(labels)), occurrence_mean["average_precision"])
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.axvline(0.5, linestyle="--", linewidth=1.0)
    axis.set(title="Held-out L5 occurrence", xlabel="Average precision")
    figure.tight_layout()
    filename = "l5_occurrence_average_precision.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    transition = path_metrics.loc[path_metrics["scope"].eq("transition")]
    transition_mean = transition.groupby(
        ["feature_set", "representation", "model"], as_index=False
    )["qlike"].mean()
    labels = (
        transition_mean["feature_set"]
        + " / "
        + transition_mean["representation"]
        + " / "
        + transition_mean["model"]
    )
    figure, axis = plt.subplots(figsize=(10.5, max(5.5, 0.32 * len(labels))))
    axis.barh(np.arange(len(labels)), transition_mean["qlike"])
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set(title="Held-out L5 transition path", xlabel="QLIKE (lower is better)")
    figure.tight_layout()
    filename = "l5_transition_path_qlike.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    local_name = "local_downside_input_paths.png"
    _plot_inputs(
        channels,
        frame,
        validation,
        LOCAL_CHANNELS,
        "Market-local downside state",
        output_dir / local_name,
    )
    outputs.append(local_name)
    global_name = "global_breadth_input_paths.png"
    _plot_inputs(
        channels,
        frame,
        validation,
        GLOBAL_CHANNELS,
        "Strictly lagged leave-one-out global breadth",
        output_dir / global_name,
    )
    outputs.append(global_name)
    return outputs


def run_downside_global_input_admission(
    *,
    evaluation_manifest: Path,
    panel_path: Path,
    output_dir: Path,
    config: DownsideGlobalAdmissionConfig = DownsideGlobalAdmissionConfig(),
) -> Path:
    config.validate()
    frame = load_evaluation_manifest(evaluation_manifest)
    frame = frame.loc[
        frame["fold"].isin(config.folds) & frame["lead"].eq(config.lead)
    ].reset_index(drop=True)
    if frame.empty or frame["fold_split"].eq("test").any():
        raise RuntimeError("assay requires non-test L5 rows")

    panel = load_ohlc_panel(panel_path)
    matrices, availability, channels = build_feature_tables(frame, panel, config)
    available = availability["downside_global_valid"].to_numpy(dtype=bool)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    candidates_dir = output / "selection_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=False)

    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = frame[list(HAR_COLUMNS)].to_numpy(dtype=float)
    causal_har = causal_har_path(frame)
    labels = frame["label"].to_numpy(dtype=int)
    occurrence_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    validation_union = np.zeros(len(frame), dtype=bool)

    for fold in config.folds:
        fold_mask = frame["fold"].eq(fold).to_numpy()
        train = fold_mask & frame["fold_split"].eq("train").to_numpy() & available
        validation = fold_mask & frame["fold_split"].eq("val").to_numpy() & available
        prequential = train & frame["prequential_valid"].to_numpy(dtype=bool)
        if not train.any() or not validation.any() or not prequential.any():
            raise RuntimeError(f"fold {fold}: missing usable train/validation rows")
        validation_union |= validation
        prediction_frame = frame.loc[
            validation,
            ["fold", "sample_id", "lead", "label", "ticker", "origin_date"],
        ].reset_index(drop=True)

        for feature_set in FEATURE_SETS:
            for representation in REPRESENTATIONS:
                features = matrices[(feature_set, representation)]
                try:
                    probability, gate_selected, gate_candidates = fit_select_occurrence(
                        features,
                        labels,
                        frame["origin_date"],
                        train,
                        validation,
                        config,
                    )
                    prediction, path_selected, path_candidates = fit_select_direct_path(
                        features,
                        causal_har,
                        y,
                        labels,
                        frame["origin_date"],
                        prequential,
                        validation,
                        config,
                    )
                except ValueError as error:
                    selection_rows.append(
                        {
                            "fold": int(fold),
                            "feature_set": feature_set,
                            "representation": representation,
                            "task": "failed",
                            "error": str(error),
                        }
                    )
                    continue

                score = probability[validation]
                target = labels[validation]
                occurrence_rows.append(
                    {
                        "fold": int(fold),
                        "feature_set": feature_set,
                        "representation": representation,
                        "samples": int(validation.sum()),
                        "average_precision": float(average_precision_score(target, score)),
                        "roc_auc": float(roc_auc_score(target, score)),
                        "balanced_accuracy_at_0p5": float(
                            balanced_accuracy_score(target, (score >= 0.5).astype(int))
                        ),
                        "mean_control_probability": float(score[target == 0].mean()),
                        "mean_transition_probability": float(score[target == 1].mean()),
                    }
                )
                path_rows.extend(
                    _path_metrics(
                        frame,
                        y,
                        har,
                        prediction,
                        validation,
                        fold=int(fold),
                        feature_set=feature_set,
                        representation=representation,
                    )
                )
                selection_rows.extend(
                    [
                        {
                            "fold": int(fold),
                            "feature_set": feature_set,
                            "representation": representation,
                            "task": "occurrence",
                            **gate_selected,
                        },
                        {
                            "fold": int(fold),
                            "feature_set": feature_set,
                            "representation": representation,
                            "task": "direct_path",
                            **path_selected,
                        },
                    ]
                )
                gate_candidates.to_csv(
                    candidates_dir
                    / f"fold_{fold}__{feature_set}__{representation}__gate.csv",
                    index=False,
                )
                path_candidates.to_csv(
                    candidates_dir
                    / f"fold_{fold}__{feature_set}__{representation}__path.csv",
                    index=False,
                )
                prediction_frame[f"prob__{feature_set}__{representation}"] = score
                for horizon in range(10):
                    prediction_frame[
                        f"pred_h{horizon + 1}__{feature_set}__{representation}"
                    ] = prediction[validation, horizon]
        prediction_rows.append(prediction_frame)

    occurrence = pd.DataFrame(occurrence_rows)
    path_metrics = pd.DataFrame(path_rows)
    selections = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if occurrence.empty or path_metrics.empty:
        raise RuntimeError("no candidate completed")

    occurrence_pooled = occurrence.groupby(
        ["feature_set", "representation"], as_index=False
    ).agg(
        samples=("samples", "sum"),
        average_precision=("average_precision", "mean"),
        roc_auc=("roc_auc", "mean"),
        balanced_accuracy_at_0p5=("balanced_accuracy_at_0p5", "mean"),
        mean_control_probability=("mean_control_probability", "mean"),
        mean_transition_probability=("mean_transition_probability", "mean"),
    )
    path_pooled = path_metrics.groupby(
        ["feature_set", "representation", "model", "scope"], as_index=False
    ).agg(
        samples=("samples", "sum"),
        qlike=("qlike", "mean"),
        rmse=("rmse", "mean"),
        mean_correction_vs_har=("mean_correction_vs_har", "mean"),
    )
    plots = _render_plots(
        occurrence,
        path_metrics,
        channels,
        frame,
        validation_union,
        output / "plots",
    )

    availability.to_csv(output / "availability.csv", index=False)
    occurrence.to_csv(output / "occurrence_fold_metrics.csv", index=False)
    occurrence_pooled.to_csv(output / "occurrence_pooled_metrics.csv", index=False)
    path_metrics.to_csv(output / "path_fold_metrics.csv", index=False)
    path_pooled.to_csv(output / "path_pooled_metrics.csv", index=False)
    selections.to_csv(output / "selected_hyperparameters.csv", index=False)
    predictions.to_csv(
        output / "validation_predictions.csv.gz", index=False, compression="gzip"
    )

    primary_occurrence = occurrence_pooled.loc[
        occurrence_pooled["feature_set"].eq("local_plus_global")
        & occurrence_pooled["representation"].eq("summary")
    ]
    primary_path = path_pooled.loc[
        path_pooled["feature_set"].eq("local_plus_global")
        & path_pooled["representation"].eq("summary")
        & path_pooled["model"].eq("har_input_direct_path")
        & path_pooled["scope"].eq("transition")
    ]
    har_path = path_pooled.loc[
        path_pooled["feature_set"].eq("local_plus_global")
        & path_pooled["representation"].eq("summary")
        & path_pooled["model"].eq("har")
        & path_pooled["scope"].eq("transition")
    ]
    if len(primary_occurrence) != 1 or len(primary_path) != 1 or len(har_path) != 1:
        raise RuntimeError("primary local-plus-global candidate did not complete")
    occurrence_row = primary_occurrence.iloc[0]
    path_row = primary_path.iloc[0]
    har_row = har_path.iloc[0]
    summary = {
        "status": "downside_global_input_admission_complete",
        "test_rows_used": 0,
        "vix_used": False,
        "same_date_cross_market_values_used": False,
        "target_market_excluded_from_global_breadth": True,
        "primary_feature_set": "local_plus_global / summary",
        "primary_occurrence_average_precision": float(occurrence_row["average_precision"]),
        "primary_occurrence_probability_gap": float(
            occurrence_row["mean_transition_probability"]
            - occurrence_row["mean_control_probability"]
        ),
        "primary_transition_qlike": float(path_row["qlike"]),
        "har_transition_qlike": float(har_row["qlike"]),
        "primary_transition_qlike_delta_vs_har": float(
            path_row["qlike"] - har_row["qlike"]
        ),
        "admission_rule": (
            "Admit a lineage to QRC only when held-out L5 transition probability "
            "exceeds control probability, average precision exceeds the balanced-panel "
            "0.5 reference, and transition path performance is not a fold-local artifact."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output
