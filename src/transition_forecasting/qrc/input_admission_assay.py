from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = tuple(f"target_h{h}" for h in range(1, 11))
HAR_COLUMNS = tuple(f"har_h{h}" for h in range(1, 11))
PREQ_COLUMNS = tuple(f"preq_resid_h{h}" for h in range(1, 11))
FEATURE_SETS = (
    "downside_return",
    "intraday_range",
    "overnight_return",
    "downside_plus_range",
    "all_three",
)
REPRESENTATIONS = ("summary", "sequence")
_TICKER_RE = re.compile(r"(?:^|_)(\^?[A-Za-z0-9.]+)_data_L(?:1|5|10)(?:_|$)")


@dataclass(frozen=True)
class InputAdmissionConfig:
    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    leads: tuple[int, ...] = (1, 5)
    window: int = 40
    inner_holdout_fraction: float = 0.25
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    range_zero_fraction_limit: float = 0.80
    seed: int = 20260724

    def validate(self) -> None:
        if not self.folds or not self.leads:
            raise ValueError("folds and leads must be nonempty")
        if self.window < 5:
            raise ValueError("window must be at least 5")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not self.alphas or any(value <= 0 for value in self.alphas):
            raise ValueError("alphas must be positive")
        if not 0.0 <= self.range_zero_fraction_limit <= 1.0:
            raise ValueError("range_zero_fraction_limit must lie in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_ticker(sample_id: str) -> str:
    match = _TICKER_RE.search(str(sample_id))
    if match is None:
        raise ValueError(f"cannot parse ticker from sample_id={sample_id!r}")
    return match.group(1)


def load_evaluation_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "fold",
        "sample_id",
        "fold_split",
        "lead",
        "label",
        "origin_date",
        "prequential_valid",
        *TARGET_COLUMNS,
        *HAR_COLUMNS,
        *PREQ_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"evaluation manifest missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("evaluation manifest is empty")
    frame = frame.copy()
    frame["ticker"] = frame["sample_id"].map(parse_ticker)
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], utc=True).dt.normalize()
    frame["prequential_valid"] = (
        frame["prequential_valid"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    if frame["prequential_valid"].isna().any():
        raise ValueError("prequential_valid contains non-boolean values")
    if frame["fold_split"].eq("test").any():
        raise RuntimeError("input admission assay must not receive test rows")
    if frame.duplicated(["fold", "sample_id"]).any():
        raise ValueError("evaluation manifest has duplicate fold/sample_id rows")
    return frame.sort_values(["fold", "origin_date", "sample_id"]).reset_index(drop=True)


def load_ohlc_panel(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(
        path,
        usecols=["date", "open", "high", "low", "close", "ticker"],
    )
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(
        ["ticker", "date"], keep="last"
    )
    output: dict[str, pd.DataFrame] = {}
    for ticker, group in frame.groupby("ticker", sort=True):
        local = group.set_index("date")[["open", "high", "low", "close"]].sort_index()
        output[str(ticker)] = local
    return output


def _summary(values: np.ndarray, prefix: str) -> tuple[np.ndarray, tuple[str, ...]]:
    data = np.asarray(values, dtype=float)
    windows = (5, 20, len(data))
    features: list[float] = [float(data[-1])]
    names = [f"{prefix}_last"]
    for width in windows:
        local = data[-min(width, len(data)) :]
        features.extend(
            [
                float(local.mean()),
                float(local.std()),
                float(local.min()),
                float(local.max()),
                float(local.sum()),
            ]
        )
        names.extend(
            [
                f"{prefix}_mean{width}",
                f"{prefix}_std{width}",
                f"{prefix}_min{width}",
                f"{prefix}_max{width}",
                f"{prefix}_sum{width}",
            ]
        )
    return np.asarray(features, dtype=float), tuple(names)


def extract_market_window(
    panel: dict[str, pd.DataFrame],
    ticker: str,
    origin_date: pd.Timestamp,
    *,
    window: int,
    range_zero_fraction_limit: float,
) -> dict[str, np.ndarray] | None:
    if ticker not in panel:
        return None
    local = panel[ticker].loc[:origin_date].dropna(subset=["close"]).tail(window + 1)
    if len(local) != window + 1:
        return None
    values = local[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return None
    open_, high, low, close = values.T
    if np.any(high < low):
        return None
    log_close = np.log(close)
    close_return = np.diff(log_close)
    downside = np.minimum(close_return, 0.0)
    intraday_range = np.log(high[1:] / low[1:])
    overnight = np.log(open_[1:] / close[:-1])
    if not (
        len(downside) == len(intraday_range) == len(overnight) == window
        and np.isfinite(downside).all()
        and np.isfinite(intraday_range).all()
        and np.isfinite(overnight).all()
    ):
        return None
    range_valid = float(
        np.mean(np.isclose(intraday_range, 0.0, atol=1e-14))
    ) <= float(range_zero_fraction_limit)
    return {
        "downside_return": downside,
        "intraday_range": intraday_range,
        "overnight_return": overnight,
        "range_valid": np.asarray([range_valid], dtype=bool),
    }


def build_feature_tables(
    manifest: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: InputAdmissionConfig,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    pd.DataFrame,
    dict[tuple[str, str], tuple[str, ...]],
]:
    config.validate()
    raw = {
        "downside_return": np.full(
            (len(manifest), config.window), np.nan, dtype=float
        ),
        "intraday_range": np.full(
            (len(manifest), config.window), np.nan, dtype=float
        ),
        "overnight_return": np.full(
            (len(manifest), config.window), np.nan, dtype=float
        ),
    }
    base_valid = np.zeros(len(manifest), dtype=bool)
    range_valid = np.zeros(len(manifest), dtype=bool)
    for row_index, row in manifest.iterrows():
        extracted = extract_market_window(
            panel,
            str(row["ticker"]),
            pd.Timestamp(row["origin_date"]),
            window=config.window,
            range_zero_fraction_limit=config.range_zero_fraction_limit,
        )
        if extracted is None:
            continue
        for name in raw:
            raw[name][row_index] = extracted[name]
        base_valid[row_index] = True
        range_valid[row_index] = bool(extracted["range_valid"][0])

    matrices: dict[tuple[str, str], np.ndarray] = {}
    names: dict[tuple[str, str], tuple[str, ...]] = {}
    components = {
        "downside_return": ("downside_return",),
        "intraday_range": ("intraday_range",),
        "overnight_return": ("overnight_return",),
        "downside_plus_range": ("downside_return", "intraday_range"),
        "all_three": (
            "downside_return",
            "intraday_range",
            "overnight_return",
        ),
    }
    for feature_set, channels in components.items():
        sequence = np.concatenate([raw[channel] for channel in channels], axis=1)
        matrices[(feature_set, "sequence")] = sequence
        names[(feature_set, "sequence")] = tuple(
            f"{channel}_t{step - config.window + 1}"
            for channel in channels
            for step in range(config.window)
        )
        summary_rows: list[np.ndarray] = []
        summary_names: tuple[str, ...] | None = None
        for row_index in range(len(manifest)):
            pieces: list[np.ndarray] = []
            local_names: list[str] = []
            for channel in channels:
                values, current_names = _summary(raw[channel][row_index], channel)
                pieces.append(values)
                local_names.extend(current_names)
            summary_rows.append(np.concatenate(pieces))
            if summary_names is None:
                summary_names = tuple(local_names)
        matrices[(feature_set, "summary")] = np.vstack(summary_rows)
        names[(feature_set, "summary")] = summary_names or tuple()

    availability = manifest[
        ["fold", "sample_id", "lead", "label", "ticker", "origin_date"]
    ].copy()
    availability["base_ohlc_valid"] = base_valid
    availability["range_valid"] = range_valid
    for feature_set in FEATURE_SETS:
        needs_range = "range" in feature_set or feature_set == "all_three"
        availability[f"valid__{feature_set}"] = base_valid & (
            range_valid if needs_range else True
        )
    return matrices, availability, names


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_logvar = np.clip(2.0 * np.asarray(y_true, dtype=float), -40.0, 20.0)
    pred_logvar = np.clip(2.0 * np.asarray(y_pred, dtype=float), -40.0, 20.0)
    ratio = np.exp(np.clip(true_logvar - pred_logvar, -40.0, 40.0))
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rmse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def chronological_inner_split(
    dates: pd.Series,
    eligible: np.ndarray,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    date_values = pd.to_datetime(dates, utc=True)
    mask = np.asarray(eligible, dtype=bool)
    unique_dates = np.asarray(sorted(date_values[mask].unique()))
    if len(unique_dates) < 4:
        raise ValueError("not enough unique dates for chronological inner split")
    holdout_count = max(1, int(np.ceil(len(unique_dates) * holdout_fraction)))
    holdout_count = min(holdout_count, len(unique_dates) - 2)
    cutoff = unique_dates[-holdout_count]
    fit = mask & (date_values < cutoff).to_numpy()
    tune = mask & (date_values >= cutoff).to_numpy()
    if fit.sum() < 5 or tune.sum() < 2:
        raise ValueError(f"inner split too small: fit={fit.sum()}, tune={tune.sum()}")
    return fit, tune


def fit_select_signed_residual(
    features: np.ndarray,
    residuals: np.ndarray,
    har: np.ndarray,
    targets: np.ndarray,
    origin_dates: pd.Series,
    fit_eligible: np.ndarray,
    prediction_eligible: np.ndarray,
    config: InputAdmissionConfig,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    x = np.asarray(features, dtype=float)
    residual = np.asarray(residuals, dtype=float)
    baseline = np.asarray(har, dtype=float)
    y = np.asarray(targets, dtype=float)
    eligible = np.asarray(fit_eligible, dtype=bool) & np.isfinite(x).all(axis=1)
    inner_fit, inner_tune = chronological_inner_split(
        origin_dates,
        eligible,
        config.inner_holdout_fraction,
    )
    rows: list[dict[str, float]] = []
    for alpha in config.alphas:
        scaler = StandardScaler().fit(x[inner_fit])
        design = scaler.transform(x)
        model = Ridge(alpha=float(alpha), fit_intercept=False).fit(
            design[inner_fit], residual[inner_fit]
        )
        prediction = baseline + model.predict(design)
        rows.append(
            {
                "alpha": float(alpha),
                "inner_rmse": rmse_loss(y[inner_tune], prediction[inner_tune]),
                "inner_qlike": qlike_loss(y[inner_tune], prediction[inner_tune]),
            }
        )
    selected = min(
        rows,
        key=lambda row: (row["inner_rmse"], row["inner_qlike"], row["alpha"]),
    )
    scaler = StandardScaler().fit(x[eligible])
    design = scaler.transform(x)
    model = Ridge(alpha=float(selected["alpha"]), fit_intercept=False).fit(
        design[eligible], residual[eligible]
    )
    correction = np.full_like(baseline, np.nan, dtype=float)
    valid_prediction = np.asarray(prediction_eligible, dtype=bool) & np.isfinite(x).all(
        axis=1
    )
    correction[valid_prediction] = model.predict(design[valid_prediction])
    prediction = baseline + correction
    selected_payload = {
        **selected,
        "inner_fit_rows": float(inner_fit.sum()),
        "inner_tune_rows": float(inner_tune.sum()),
        "full_fit_rows": float(eligible.sum()),
    }
    return prediction, selected_payload, pd.DataFrame(rows)


def metric_rows(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    har: np.ndarray,
    validation: np.ndarray,
    *,
    fold: int,
    lead: int,
    feature_set: str,
    representation: str,
) -> list[dict[str, object]]:
    scopes = {
        "all": validation,
        "control": validation & frame["label"].eq(0).to_numpy(),
        "transition": validation & frame["label"].eq(1).to_numpy(),
    }
    rows: list[dict[str, object]] = []
    common_available = np.isfinite(prediction).all(axis=1) & np.isfinite(har).all(axis=1)
    for model_name, forecast in (("har", har), ("har_plus_input", prediction)):
        for scope, mask in scopes.items():
            valid = np.asarray(mask, dtype=bool) & common_available
            if not valid.any():
                continue
            rows.append(
                {
                    "fold": int(fold),
                    "lead": int(lead),
                    "feature_set": feature_set,
                    "representation": representation,
                    "model": model_name,
                    "scope": scope,
                    "samples": int(valid.sum()),
                    "qlike": qlike_loss(y[valid], forecast[valid]),
                    "rmse": rmse_loss(y[valid], forecast[valid]),
                    "mean_correction": float(np.mean(forecast[valid] - har[valid])),
                }
            )
    return rows


def run_input_admission_assay(
    *,
    evaluation_manifest: Path,
    panel_path: Path,
    output_dir: Path,
    config: InputAdmissionConfig = InputAdmissionConfig(),
) -> Path:
    config.validate()
    frame = load_evaluation_manifest(evaluation_manifest)
    panel = load_ohlc_panel(panel_path)
    matrices, availability, feature_names = build_feature_tables(frame, panel, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = frame[list(HAR_COLUMNS)].to_numpy(dtype=float)
    residual = frame[list(PREQ_COLUMNS)].to_numpy(dtype=float)
    metric_output: list[dict[str, object]] = []
    selection_output: list[dict[str, object]] = []
    candidate_output: list[pd.DataFrame] = []
    prediction_output: list[pd.DataFrame] = []

    for fold in config.folds:
        fold_rows = frame["fold"].eq(fold).to_numpy()
        for lead in config.leads:
            local = fold_rows & frame["lead"].eq(lead).to_numpy()
            train = local & frame["fold_split"].eq("train").to_numpy()
            validation = local & frame["fold_split"].eq("val").to_numpy()
            prequential = train & frame["prequential_valid"].to_numpy()
            if not train.any() or not validation.any() or not prequential.any():
                raise RuntimeError(
                    f"fold={fold}, lead={lead}: missing train/validation/prequential rows"
                )
            for feature_set in FEATURE_SETS:
                available = availability[f"valid__{feature_set}"].to_numpy(dtype=bool)
                for representation in REPRESENTATIONS:
                    features = matrices[(feature_set, representation)]
                    fit_eligible = prequential & available
                    prediction_eligible = local & available
                    try:
                        prediction, selected, candidates = fit_select_signed_residual(
                            features,
                            residual,
                            har,
                            y,
                            frame["origin_date"],
                            fit_eligible,
                            prediction_eligible,
                            config,
                        )
                    except ValueError as error:
                        selection_output.append(
                            {
                                "fold": int(fold),
                                "lead": int(lead),
                                "feature_set": feature_set,
                                "representation": representation,
                                "status": "skipped",
                                "reason": str(error),
                            }
                        )
                        continue
                    selection_output.append(
                        {
                            "fold": int(fold),
                            "lead": int(lead),
                            "feature_set": feature_set,
                            "representation": representation,
                            "status": "complete",
                            **selected,
                            "feature_width": int(features.shape[1]),
                        }
                    )
                    candidates.insert(0, "fold", int(fold))
                    candidates.insert(1, "lead", int(lead))
                    candidates.insert(2, "feature_set", feature_set)
                    candidates.insert(3, "representation", representation)
                    candidate_output.append(candidates)
                    metric_output.extend(
                        metric_rows(
                            frame,
                            y,
                            prediction,
                            har,
                            validation,
                            fold=int(fold),
                            lead=int(lead),
                            feature_set=feature_set,
                            representation=representation,
                        )
                    )
                    selected_rows = frame.loc[validation].copy()
                    selected_rows["feature_set"] = feature_set
                    selected_rows["representation"] = representation
                    for horizon in range(10):
                        selected_rows[f"pred_h{h + 1}"] = prediction[validation, horizon]
                    prediction_output.append(selected_rows)

    metrics = pd.DataFrame(metric_output)
    selections = pd.DataFrame(selection_output)
    candidates = (
        pd.concat(candidate_output, ignore_index=True)
        if candidate_output
        else pd.DataFrame()
    )
    predictions = (
        pd.concat(prediction_output, ignore_index=True)
        if prediction_output
        else pd.DataFrame()
    )
    availability.to_csv(output / "availability.csv", index=False)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    selections.to_csv(output / "selected_alphas.csv", index=False)
    candidates.to_csv(output / "inner_candidates.csv", index=False)
    predictions.to_csv(
        output / "validation_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    complete = metrics.loc[metrics["scope"].eq("all")]
    pivot = complete.pivot_table(
        index=["fold", "lead", "feature_set", "representation"],
        columns="model",
        values=["qlike", "rmse"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    if not pivot.empty:
        pivot["delta_qlike"] = (
            pivot["qlike__har_plus_input"] - pivot["qlike__har"]
        )
        pivot["delta_rmse"] = pivot["rmse__har_plus_input"] - pivot["rmse__har"]
        pivot.to_csv(output / "paired_fold_deltas.csv", index=False)
        summary_table = (
            pivot.groupby(["lead", "feature_set", "representation"], as_index=False)
            .agg(
                folds=("fold", "nunique"),
                mean_delta_qlike=("delta_qlike", "mean"),
                mean_delta_rmse=("delta_rmse", "mean"),
                qlike_wins=("delta_qlike", lambda values: int((values < 0).sum())),
                rmse_wins=("delta_rmse", lambda values: int((values < 0).sum())),
            )
            .sort_values(["lead", "mean_delta_rmse", "mean_delta_qlike"])
        )
    else:
        summary_table = pd.DataFrame()
    summary_table.to_csv(output / "admission_summary.csv", index=False)

    summary = {
        "status": "input_admission_assay_complete",
        "test_rows_used": 0,
        "config": config.to_dict(),
        "evaluation_rows": int(len(frame)),
        "tickers": int(frame["ticker"].nunique()),
        "base_ohlc_valid_rows": int(availability["base_ohlc_valid"].sum()),
        "range_valid_rows": int(availability["range_valid"].sum()),
        "interpretation": (
            "This is an input-admission screen. Models are no-intercept signed residual "
            "probes selected by chronological inner RMSE. A channel should not be sent "
            "through QRC unless it improves the requested lead across folds without "
            "relying on a universal offset."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "feature_names.json").write_text(
        json.dumps(
            {
                f"{key[0]}__{key[1]}": list(value)
                for key, value in feature_names.items()
            },
            indent=2,
        )
        + "\n"
    )
    return output
