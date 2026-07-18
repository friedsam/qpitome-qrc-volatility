from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.cross_market_mapping import MARKET_GROUP_TO_TICKER


def load_stage_d(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(run_dir / "sample_manifest.csv").reset_index(drop=True)
    with np.load(run_dir / "sequence_tensors.npz", allow_pickle=True) as data:
        sequences = np.asarray(data["X"], dtype=float)
        tensor_ids = data["sample_id"].astype(str)
    if not np.array_equal(manifest["sample_id"].astype(str).to_numpy(), tensor_ids):
        raise ValueError("manifest and tensor sample IDs are not aligned")
    return manifest, sequences[..., 0]


def build_signals(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for ticker, group in panel.groupby("ticker", sort=True):
        frame = group[["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.normalize()
        frame = frame.sort_values("date").drop_duplicates("date")
        close = pd.to_numeric(frame["close"], errors="coerce")
        log_return = np.log(close).diff()
        signal = pd.DataFrame({"date": frame["date"]})
        signal["abs_log_return"] = log_return.abs()
        signal["log_abs_log_return"] = np.log(log_return.abs().clip(lower=1e-12))
        for window in (5, 10, 20):
            rv = log_return.rolling(window, min_periods=window).std(ddof=0)
            signal[f"rv{window}"] = rv
            signal[f"log_rv{window}"] = np.log(rv.clip(lower=1e-12))
            mean_abs = log_return.abs().rolling(window, min_periods=window).mean()
            signal[f"mean_abs{window}"] = mean_abs
            signal[f"log_mean_abs{window}"] = np.log(mean_abs.clip(lower=1e-12))
        output[str(ticker)] = signal.set_index("date")
    return output


def _extract_window(signal: pd.Series, origin: pd.Timestamp, length: int = 40) -> np.ndarray | None:
    history = signal.loc[:origin].dropna().tail(length)
    if len(history) != length:
        return None
    return history.to_numpy(dtype=float)


def compare_transforms(
    manifest: pd.DataFrame,
    stage_d_sequences: np.ndarray,
    signals: dict[str, pd.DataFrame],
    *,
    max_samples: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = [column for frame in signals.values() for column in frame.columns]
    candidates = sorted(set(candidates))
    subset = manifest if max_samples is None else manifest.head(max_samples)

    for candidate in candidates:
        paired_stage = []
        paired_raw = []
        matched_samples = 0
        for index, row in subset.iterrows():
            ticker = MARKET_GROUP_TO_TICKER.get(str(row["market_group"]))
            if ticker is None or ticker not in signals or candidate not in signals[ticker]:
                continue
            origin = pd.to_datetime(row["origin_date"], utc=True).normalize()
            raw_window = _extract_window(signals[ticker][candidate], origin)
            if raw_window is None:
                continue
            paired_stage.append(stage_d_sequences[index])
            paired_raw.append(raw_window)
            matched_samples += 1
        if not paired_stage:
            continue
        stage = np.concatenate(paired_stage)
        raw = np.concatenate(paired_raw)
        finite = np.isfinite(stage) & np.isfinite(raw)
        stage = stage[finite]
        raw = raw[finite]
        if len(stage) < 3:
            continue
        correlation = float(np.corrcoef(stage, raw)[0, 1])
        slope, intercept = np.polyfit(raw, stage, deg=1)
        fitted = intercept + slope * raw
        rmse = float(np.sqrt(np.mean((stage - fitted) ** 2)))
        mae = float(np.mean(np.abs(stage - fitted)))
        rows.append({
            "candidate": candidate,
            "matched_samples": matched_samples,
            "paired_points": int(len(stage)),
            "correlation": correlation,
            "affine_slope": float(slope),
            "affine_intercept": float(intercept),
            "affine_rmse": rmse,
            "affine_mae": mae,
        })
        print(
            f"{candidate:18s} samples={matched_samples:4d} corr={correlation: .6f} "
            f"rmse={rmse:.6f}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values(["correlation", "affine_rmse"], ascending=[False, True])


def mapping_audit(manifest: pd.DataFrame, panel_tickers: set[str]) -> pd.DataFrame:
    rows = []
    for market_group in sorted(manifest["market_group"].astype(str).unique()):
        ticker = MARKET_GROUP_TO_TICKER.get(market_group)
        rows.append({
            "market_group": market_group,
            "mapped_ticker": ticker,
            "mapping_present": ticker is not None,
            "ticker_in_panel": ticker in panel_tickers if ticker is not None else False,
            "samples": int(manifest["market_group"].astype(str).eq(market_group).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify which raw-price volatility transform reproduces Stage D input windows.")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/raw/transition_forecasting/global_stock_indices_historical_data/all_indices_data.csv"),
    )
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/cross_market_transform_match"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, stage_d_sequences = load_stage_d(args.stage_d_run)
    panel = pd.read_csv(args.panel, usecols=["date", "close", "ticker"])
    audit = mapping_audit(manifest, set(panel["ticker"].astype(str).unique()))
    audit.to_csv(run_dir / "market_group_ticker_mapping.csv", index=False)
    if not audit["ticker_in_panel"].all():
        missing = audit.loc[~audit["ticker_in_panel"], ["market_group", "mapped_ticker"]]
        raise ValueError(f"incomplete market mapping:\n{missing.to_string(index=False)}")

    signals = build_signals(panel)
    results = compare_transforms(
        manifest,
        stage_d_sequences,
        signals,
        max_samples=args.max_samples,
    )
    results.to_csv(run_dir / "transform_match_results.csv", index=False)
    best = results.iloc[0].to_dict()
    payload = {
        "test_evaluated": False,
        "mapped_markets": int(audit["ticker_in_panel"].sum()),
        "manifest_markets": int(len(audit)),
        "best_transform": best,
        "interpretation": "High correlation plus low affine error indicates the raw transform underlying Stage D.",
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nTRANSFORM RANKING", flush=True)
    print(results.to_string(index=False), flush=True)
    print("\n" + json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
