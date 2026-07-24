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
    downside = negative.rolling(window, min_periods=max(2, window // 2)).sum()
    total = (negative + positive).rolling(
        window, min_periods=max(2, window // 2)
    ).sum()
    return downside / total.where(total > 1e-16)


def build_local_state(
    panel: dict[str, pd.DataFrame],
    config: DownsideGlobalAdmissionConfig,
) -> dict[str, pd.DataFrame]:
    """Construct market-relative, causal downside-state channels."""

    output: dict[str, pd.DataFrame] = {}
    for ticker, values in panel.items():
        local = values.dropna(subset=["close"]).copy()
        close = pd.to_numeric(local["close"], errors="coerce").where(lambda x: x > 0)
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
        state = pd.DataFrame(
            {
                "local_downside_z": standardized.clip(upper=0.0),
                "local_drawdown_60": log_close - rolling_peak,
                "local_downside_share_5": _rolling_downside_share(
                    returns, config.semivariance_window
                ),
            },
            index=local.index,
        )
        output[str(ticker)] = state.replace([np.inf, -np.inf], np.nan)
    return output


def _wide_standardized_returns(
    panel: dict[str, pd.DataFrame],
    config: DownsideGlobalAdmissionConfig,
) -> pd.DataFrame:
    returns: dict[str, pd.Series] = {}
    for ticker, values in panel.items():
        if str(ticker) == "^VIX":
            continue
        close = pd.to_numeric(values["close"], errors="coerce").where(lambda x: x > 0)
        local_return = np.log(close).diff()
        prior_scale = (
            local_return.shift(1)
            .rolling(
                config.local_volatility_lookback,
                min_periods=max(5, config.local_volatility_lookback // 2),
            )
            .std()
        )
        returns[str(ticker)] = local_return / prior_scale.where(prior_scale > 1e-12)
    return pd.DataFrame(returns).sort_index().replace([np.inf, -np.inf], np.nan)


def build_leave_one_out_global_breadth(
    panel: dict[str, pd.DataFrame],
    config: DownsideGlobalAdmissionConfig,
) -> dict[str, pd.DataFrame]:
    """Build strictly prior-date global breadth, excluding the target market and VIX."""

    standardized = _wide_standardized_returns(panel, config)
    output: dict[str, pd.DataFrame] = {}
    for target_ticker in panel:
        peers = standardized.drop(columns=[str(target_ticker)], errors="ignore")
        count = peers.notna().sum(axis=1)
        median = peers.median(axis=1, skipna=True)
        dispersion = peers.sub(median, axis=0).abs().median(axis=1, skipna=True)
        breadth = pd.DataFrame(
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
        # Daily files do not preserve exchange-close timestamps. A full union-date
        # lag is therefore mandatory: no value dated on the target origin date can
        # enter that origin's feature window.
        output[str(target_ticker)] = breadth.shift(1)
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
    origin = pd.Timestamp(origin_date)
    local = local_state[ticker].loc[:origin].dropna().tail(config.window)
    if len(local) != config.window:
        return None
    dates = local.index
    global_window = global_state[ticker].reindex(dates)
    required_global = list(GLOBAL_CHANNELS) + ["global_peer_count"]
    if global_window[required_global].isna().any().any():
        return None
    if (global_window["global_peer_count"] < config.minimum_peer_markets).any():
        return None
    result = {
        channel: local[channel].to_numpy(dtype=float) for channel in LOCAL_CHANNELS
    }
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


def _compact_summary(values: np.ndarray, prefix: str) -> tuple[np.ndarray, tuple[str, ...]]:
    data = np.asarray(values, dtype=float)
    payload = [float(data[-1])]
    names = [f"{prefix}_last"]
    for width in (5, 20, len(data)):
        local = data[-min(width, len(data)) :]
        payload.extend(
            [
                float(local.mean()),
                float(local.std()),
                float(local.min()),
                float(local.max()),
            ]
        )
        names.extend(
            [
                f"{prefix}_mean{width}",
                f"{prefix}_std{width}",
                f"{prefix}_min{width}",
                f"{prefix}_max{width}",
            ]
        )
    return np.asarray(payload, dtype=float), tuple(names)


def build_feature_tables(
    manifest: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: DownsideGlobalAdmissionConfig,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    config.validate()
    local_state = build_local_state(panel, config)
    global_state = build_leave_one_out_global_breadth(panel, config)
    channel_values = {
        channel: np.full((len(manifest), config.window), np.nan, dtype=float)
        for channel in LOCAL_CHANNELS + GLOBAL_CHANNELS
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
        for channel, values in extracted.items():
            channel_values[channel][row_index] = values
        valid[row_index] = True

    matrices: dict[tuple[str, str], np.ndarray] = {}
    for feature_set, channels in FEATURE_SETS.items():
        matrices[(feature_set, "sequence")] = np.concatenate(
            [channel_values[channel] for channel in channels], axis=1
        )
        summary_rows: list[np.ndarray] = []
        for row_index in range(len(manifest)):
            pieces = [
                _compact_summary(channel_values[channel][row_index], channel)[0]
                for channel in channels
            ]
            summary_rows.append(np.concatenate(pieces))
        matrices[(feature_set, "summary")] = np.vstack(summary_rows)

    availability = manifest[
        ["fold", "sample_id", "lead", "label", "ticker", "origin_date"]
    ].copy()
    availability["downside_global_valid"] = valid
    return matrices, availability, channel_values


def _causal_har_path(frame: pd.DataFrame) -> np.ndarray:
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
    target = np.asarray(labels, dtype=int)
    train = np.asarray(train_mask, dtype=bool) & np.isfinite(x).all(axis=1)
    validation = np.asarray(validation_mask, dtype=bool) & np.isfinite(x).all(axis=1)
    inner_fit, inner_tune = chronological_inner_split(
        dates, train, config.inner_holdout_fraction
    )
    for name, mask in (("inner_fit", inner_fit), ("inner_tune", inner_tune), ("train", train)):
        if set(np.unique(target[mask])) != {0, 1}:
            raise ValueError(f"{name} must contain both L5 classes")
    rows: list[dict[str, float]] = []
    for c_value in config.gate_cs:
        scaler = StandardScaler().fit(x[inner_fit])
        design = scaler.transform(x)
        model = LogisticRegression(
            C=float(c_value),
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ).fit(design[inner_fit], target[inner_fit])
        probability = model.predict_proba(design)[:, 1]
        rows.append(
            {
                "c_value": float(c_value),
                "inner_average_precision": float(
                    average_precision_score(target[inner_tune], probability[inner_tune])
                ),
                "inner_roc_auc": float(
                    roc_auc_score(target[inner_tune], probability[inner_tune])
                ),
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
    design = scaler.transform(x)
    model = LogisticRegression(
        C=float(selected["c_value"]),
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    ).fit(design[train], target[train])
    probability = np.full(len(x), np.nan, dtype=float)
    probability[validation] = model.predict_proba(design[validation])[:, 1]
    return probability, dict(selected), pd.DataFrame(rows)


def fit_select_direct_path(
    input_features: np.ndarray,
    causal_har: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    dates: pd.Series,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: DownsideGlobalAdmissionConfig,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    x = np.asarray(input_features, dtype=float)
    har = np.asarray(causal_har, dtype=float)
    y = np.asarray(targets, dtype=float)
    design_raw = np.column_stack([har, x])
    eligible = np.asarray(fit_mask, dtype=bool) & np.isfinite(design_raw).all(axis=1)
    validation = np.asarray(validation_mask, dtype=bool) & np.isfinite(design_raw).all(axis=1)
    inner_fit, inner_tune = chronological_inner_split(
        dates, eligible, config.inner_holdout_fraction
    )
    tune_transition = inner_tune & (np.asarray(labels, dtype=int) == 1)
    if not tune_transition.any():
        raise ValueError("inner path holdout contains no L5 transitions")
    rows: list[dict[str, float]] = []
    for alpha in config.path_alphas:
        scaler = StandardScaler().fit(design_raw[inner_fit])
        design = scaler.transform(design_raw)
        model = Ridge(alpha=float(alpha)).fit(design[inner_fit], y[inner_fit])
        prediction = model.predict(design)
        rows.append(
            {
                "alpha": float(alpha),
                "inner_transition_qlike": qlike_loss(
                    y[tune_transition], prediction[tune_transition]
                ),
                "inner_overall_rmse": rmse_loss(y[inner_tune], prediction[inner_tune]),
                "inner_overall_qlike": qlike_loss(y[inner_tune], prediction[inner_tune]),
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
    scaler = StandardScaler().fit(design_raw[eligible])
    design = scaler.transform(design_raw)
    model = Ridge(alpha=float(selected["alpha"])).fit(design[eligible], y[eligible])
    prediction = np.full_like(y, np.nan, dtype=float)
    prediction[validation] = model.predict(design[validation])
    return prediction, dict(selected), pd.DataFrame(rows)


def _metric_rows(
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
    rows: list[dict[str, object]] = []
    common = np.isfinite(prediction).all(axis=1) & np.isfinite(har).all(axis=1)
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


def _render_plots(
    occurrence: pd.DataFrame,
    path_metrics: pd.DataFrame,
    channel_values: dict[str, np.ndarray],
    frame: pd.DataFrame,
    validation_rows: np.ndarray,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    pooled_occurrence = occurrence.groupby(
        ["feature_set", "representation"], as_index=False
    )[["average_precision", "roc_auc"]].mean()
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    labels = (
        pooled_occurrence["feature_set"] + " / " + pooled_occurrence["representation"]
    )
    axis.barh(np.arange(len(labels)), pooled_occurrence["average_precision"])
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.axvline(0.5, linestyle="--", linewidth=1.0)
    axis.set(title="Held-out L5 occurrence: average precision", xlabel="Average precision")
    figure.tight_layout()
    filename = "l5_occurrence_average_precision.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    pooled_path = path_metrics.groupby(
        ["feature_set", "representation", "model", "scope"], as_index=False
    )[["qlike", "rmse"]].mean()
    local = pooled_path.loc[pooled_path["scope"].eq("transition")].copy()
    local["label"] = (
        local["feature_set"] + " / " + local["representation"] + " / " + local["model"]
    )
    figure, axis = plt.subplots(figsize=(10.5, max(5.5, 0.32 * len(local))))
    axis.barh(np.arange(len(local)), local["qlike"])
    axis.set_yticks(np.arange(len(local)))
    axis.set_yticklabels(local["label"], fontsize=8)
    axis.set(title="Held-out L5 transition path QLIKE", xlabel="QLIKE (lower is better)")
    figure.tight_layout()
    filename = "l5_transition_path_qlike.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    validation_frame = frame.loc[validation_rows].reset_index(drop=True)
    for channels, title, filename in (
        (LOCAL_CHANNELS, "Market-local downside state", "local_downside_input_paths.png"),
        (GLOBAL_CHANNELS, "Strictly lagged leave-one-out global breadth", "global_breadth_input_paths.png"),
    ):
        figure, axes = plt.subplots(len(channels), 1, figsize=(9.0, 2.8 * len(channels)), sharex=True)
        axes_array = np.atleast_1d(axes)
        for axis, channel in zip(axes_array, channels, strict=True):
            values = channel_values[channel][validation_rows]
            for label, name in ((0, "controls"), (1, "transitions")):
                mask = validation_frame["label"].eq(label).to_numpy()
                if mask.any():
                    axis.plot(
                        np.arange(-values.shape[1] + 1, 1),
                        np.nanmean(values[mask], axis=0),
                        label=name,
                    )
            axis.axvline(0, linestyle="--", linewidth=1.0)
            axis.set_ylabel(channel)
            axis.grid(alpha=0.25)
        axes_array[0].set_title(title)
        axes_array[-1].set_xlabel("Trading days before origin")
        axes_array[0].legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)
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
        raise RuntimeError("admission assay requires non-test L5 rows")
    panel = load_ohlc_panel(panel_path)
    matrices, availability, channel_values = build_feature_tables(frame, panel, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = frame[list(HAR_COLUMNS)].to_numpy(dtype=float)
    causal_har = _causal_har_path(frame)
    labels = frame["label"].to_numpy(dtype=int)
    available = availability["downside_global_valid"].to_numpy(dtype=bool)

    occurrence_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    validation_union = np.zeros(len(frame), dtype=bool)

    candidate_dir = output / "selection_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    for fold in config.folds:
        fold_rows = frame["fold"].eq(fold).to_numpy()
        train = fold_rows & frame["fold_split"].eq("train").to_numpy() & available
        validation = fold_rows & frame["fold_split"].eq("val").to_numpy() & available
        prequential = train & frame["prequential_valid"].to_numpy(dtype=bool)
        if not train.any() or not validation.any() or not prequential.any():
            raise RuntimeError(f"fold {fold}: missing usable train/validation rows")
        validation_union |= validation
        local_prediction = frame.loc[validation, [
            "fold", "sample_id", "lead", "label", "ticker", "origin_date"
        ]].reset_index(drop=True)

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

                target = labels[validation]
                score = probability[validation]
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
                    _metric_rows(
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
                    candidate_dir / f"fold_{fold}__{feature_set}__{representation}__gate.csv",
                    index=False,
                )
                path_candidates.to_csv(
                    candidate_dir / f"fold_{fold}__{feature_set}__{representation}__path.csv",
                    index=False,
                )
                local_prediction[
                    f"prob__{feature_set}__{representation}"
                ] = score
                for horizon in range(10):
                    local_prediction[
                        f"pred_h{horizon + 1}__{feature_set}__{representation}"
                    ] = prediction[validation, horizon]
        prediction_rows.append(local_prediction)

    occurrence = pd.DataFrame(occurrence_rows)
    path_metrics = pd.DataFrame(path_rows)
    selections = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if occurrence.empty or path_metrics.empty:
        raise RuntimeError("no admission candidates completed")

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
        channel_values,
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
    ].iloc[0]
    primary_path = path_pooled.loc[
        path_pooled["feature_set"].eq("local_plus_global")
        & path_pooled["representation"].eq("summary")
        & path_pooled["model"].eq("har_input_direct_path")
        & path_pooled["scope"].eq("transition")
    ].iloc[0]
    har_path = path_pooled.loc[
        path_pooled["feature_set"].eq("local_plus_global")
        & path_pooled["representation"].eq("summary")
        & path_pooled["model"].eq("har")
        & path_pooled["scope"].eq("transition")
    ].iloc[0]
    summary = {
        "status": "downside_global_input_admission_complete",
        "test_rows_used": 0,
        "vix_used": False,
        "same_date_cross_market_values_used": False,
        "target_market_excluded_from_global_breadth": True,
        "primary_feature_set": "local_plus_global / summary",
        "primary_occurrence_average_precision": float(
            primary_occurrence["average_precision"]
        ),
        "primary_occurrence_probability_gap": float(
            primary_occurrence["mean_transition_probability"]
            - primary_occurrence["mean_control_probability"]
        ),
        "primary_transition_qlike": float(primary_path["qlike"]),
        "har_transition_qlike": float(har_path["qlike"]),
        "primary_transition_qlike_delta_vs_har": float(
            primary_path["qlike"] - har_path["qlike"]
        ),
        "admission_rule": (
            "Admit an input lineage to QRC only if later-fold L5 occurrence scores "
            "are higher for transitions than controls, average precision exceeds the "
            "balanced-panel 0.5 reference, and the direct HAR-plus-input path does not "
            "degrade transition QLIKE across most folds."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output
