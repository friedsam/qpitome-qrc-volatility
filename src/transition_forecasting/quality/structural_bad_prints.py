from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructuralBadPrintPolicy:
    min_log_range: float = 0.20
    max_body_log_move: float = float(np.log(1.05))
    min_wick_log_excursion: float = float(np.log(1.20))
    rolling_window: int = 252
    min_periods: int = 60
    mad_multiplier: float = 12.0
    mad_consistency_constant: float = 1.4826

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _rolling_mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def normalize_ohlc(frame: pd.DataFrame, source: Path | str = "<memory>") -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"{source}: missing required columns {sorted(missing)}")

    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(
            result[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    valid = (
        result["date"].notna()
        & result["open"].gt(0)
        & result["high"].gt(0)
        & result["low"].gt(0)
        & result["close"].gt(0)
        & result["high"].ge(result["low"])
    )
    return (
        result.loc[valid]
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def flag_structural_bad_prints(
    frame: pd.DataFrame,
    *,
    index_name: str,
    policy: StructuralBadPrintPolicy = StructuralBadPrintPolicy(),
) -> pd.DataFrame:
    data = normalize_ohlc(frame, source=index_name)
    data["log_range"] = np.log(data["high"] / data["low"])
    data["body"] = np.abs(np.log(data["close"] / data["open"]))

    body_high = data[["open", "close"]].max(axis=1)
    body_low = data[["open", "close"]].min(axis=1)
    data["up_excursion"] = np.log(data["high"] / body_high)
    data["down_excursion"] = np.log(body_low / data["low"])

    prior_range = data["log_range"].shift(1)
    data["range_median_252"] = prior_range.rolling(
        policy.rolling_window,
        min_periods=policy.min_periods,
    ).median()
    data["range_mad_252"] = prior_range.rolling(
        policy.rolling_window,
        min_periods=policy.min_periods,
    ).apply(_rolling_mad, raw=True)
    robust_threshold = (
        data["range_median_252"]
        + policy.mad_multiplier
        * policy.mad_consistency_constant
        * data["range_mad_252"]
    )

    data["suspected_bad_print"] = (
        data["log_range"].gt(policy.min_log_range)
        & data["log_range"].gt(robust_threshold)
        & data["body"].lt(policy.max_body_log_move)
        & (
            data["down_excursion"].gt(policy.min_wick_log_excursion)
            | data["up_excursion"].gt(policy.min_wick_log_excursion)
        )
    )
    data.insert(0, "index", str(index_name))
    return data


def audit_directory(
    root: Path,
    *,
    policy: StructuralBadPrintPolicy = StructuralBadPrintPolicy(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[pd.DataFrame] = []
    flagged_rows: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.csv")):
        frame = pd.read_csv(path)
        audited = flag_structural_bad_prints(frame, index_name=path.stem, policy=policy)
        audited.insert(1, "source_path", str(path))
        all_rows.append(audited)
        flagged_rows.append(audited.loc[audited["suspected_bad_print"]].copy())

    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    flagged = pd.concat(flagged_rows, ignore_index=True) if flagged_rows else pd.DataFrame()
    return full, flagged


def annotate_sample_manifest(
    manifest: pd.DataFrame,
    flagged_daily_rows: pd.DataFrame,
) -> pd.DataFrame:
    result = manifest.copy()
    for column in ("origin_date", "input_start_date", "target_end_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce", utc=True).dt.tz_localize(None)

    flags = flagged_daily_rows.copy()
    flags["date"] = pd.to_datetime(flags["date"], errors="coerce", utc=True).dt.tz_localize(None)
    dates_by_index = (
        flags.groupby("index")["date"]
        .apply(lambda values: tuple(sorted(values.dropna().unique())))
        .to_dict()
    )

    def dates_in_interval(row: pd.Series, *, target: bool) -> tuple[pd.Timestamp, ...]:
        dates = dates_by_index.get(str(row["index"]), ())
        if target:
            start, end = row["origin_date"], row["target_end_date"]
            return tuple(pd.Timestamp(date) for date in dates if pd.notna(start) and pd.notna(end) and start < pd.Timestamp(date) <= end)
        start, end = row["input_start_date"], row["origin_date"]
        return tuple(pd.Timestamp(date) for date in dates if pd.notna(start) and pd.notna(end) and start <= pd.Timestamp(date) <= end)

    input_dates = result.apply(lambda row: dates_in_interval(row, target=False), axis=1)
    target_dates = result.apply(lambda row: dates_in_interval(row, target=True), axis=1)
    result["bad_print_in_input"] = input_dates.str.len().gt(0)
    result["bad_print_in_target"] = target_dates.str.len().gt(0)
    result["bad_print_any"] = result["bad_print_in_input"] | result["bad_print_in_target"]
    result["input_bad_print_dates"] = input_dates.apply(lambda values: "|".join(pd.Timestamp(value).strftime("%Y-%m-%d") for value in values))
    result["target_bad_print_dates"] = target_dates.apply(lambda values: "|".join(pd.Timestamp(value).strftime("%Y-%m-%d") for value in values))
    return result
