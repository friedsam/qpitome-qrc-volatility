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
        sequences = np.asarray(data["X"], dtype=float)[..., 0]
        tensor_ids = data["sample_id"].astype(str)
    if not np.array_equal(manifest["sample_id"].astype(str).to_numpy(), tensor_ids):
        raise ValueError("manifest and tensor sample IDs are not aligned")
    return manifest, sequences


def build_ohlc_signals(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for ticker, group in panel.groupby("ticker", sort=True):
        frame = group[["date", "open", "high", "low", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.normalize()
        frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        o, h, l, c = (frame[column].clip(lower=1e-12) for column in ("open", "high", "low", "close"))
        prev_c = c.shift(1)
        cc = np.log(c).diff()
        hl = np.log(h / l)
        co = np.log(c / o)
        oc = np.log(o / prev_c)
        gk_var = 0.5 * hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * co.pow(2)
        rs_var = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

        base = {
            "abs_close_return": cc.abs(),
            "high_low": hl.abs(),
            "parkinson": np.sqrt((hl.pow(2) / (4.0 * np.log(2.0))).clip(lower=0.0)),
            "garman_klass": np.sqrt(gk_var.clip(lower=0.0)),
            "rogers_satchell": np.sqrt(rs_var.clip(lower=0.0)),
            "true_range_pct": tr / prev_c,
            "overnight_abs": oc.abs(),
        }
        features: dict[str, pd.Series] = {}
        for column, values in base.items():
            features[column] = values
            features[f"log_{column}"] = np.log(values.clip(lower=1e-12))
            for window in (3, 5, 10, 20):
                mean = values.rolling(window, min_periods=window).mean()
                rms = np.sqrt(values.pow(2).rolling(window, min_periods=window).mean())
                features[f"{column}_mean{window}"] = mean
                features[f"log_{column}_mean{window}"] = np.log(mean.clip(lower=1e-12))
                features[f"{column}_rms{window}"] = rms
                features[f"log_{column}_rms{window}"] = np.log(rms.clip(lower=1e-12))
        output[str(ticker)] = pd.DataFrame(features, index=frame.index)
    return output


def extract_window(series: pd.Series, origin: pd.Timestamp, length: int, alignment: str) -> np.ndarray | None:
    if alignment == "previous_or_same":
        history = series.loc[:origin].dropna().tail(length)
    elif alignment == "strict_before":
        history = series.loc[series.index < origin].dropna().tail(length)
    else:
        raise ValueError(f"unknown alignment: {alignment}")
    if len(history) != length:
        return None
    return history.to_numpy(dtype=float)


def compare(
    manifest: pd.DataFrame,
    stage: np.ndarray,
    signals: dict[str, pd.DataFrame],
    *,
    max_samples: int | None = None,
) -> pd.DataFrame:
    subset = manifest if max_samples is None else manifest.head(max_samples)
    candidates = sorted({column for frame in signals.values() for column in frame.columns})
    rows: list[dict[str, object]] = []

    for alignment in ("previous_or_same", "strict_before"):
        for candidate in candidates:
            stage_parts = []
            raw_parts = []
            matched = 0
            for index, row in subset.iterrows():
                ticker = MARKET_GROUP_TO_TICKER[str(row["market_group"])]
                frame = signals.get(ticker)
                if frame is None or candidate not in frame:
                    continue
                origin = pd.to_datetime(row["origin_date"], utc=True).normalize()
                raw = extract_window(frame[candidate], origin, 40, alignment)
                if raw is None:
                    continue
                stage_parts.append(stage[index])
                raw_parts.append(raw)
                matched += 1
            if not stage_parts:
                continue
            y = np.concatenate(stage_parts)
            x = np.concatenate(raw_parts)
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            if len(x) < 3 or np.std(x) == 0.0:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            slope, intercept = np.polyfit(x, y, deg=1)
            fitted = intercept + slope * x
            rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
            rows.append({
                "alignment": alignment,
                "candidate": candidate,
                "matched_samples": matched,
                "paired_points": int(len(x)),
                "correlation": corr,
                "affine_slope": float(slope),
                "affine_intercept": float(intercept),
                "affine_rmse": rmse,
            })
            print(
                f"{alignment:16s} {candidate:34s} samples={matched:4d} corr={corr: .6f} rmse={rmse:.6f}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["correlation", "affine_rmse"], ascending=[False, True])


def main() -> None:
    parser = argparse.ArgumentParser(description="Match Stage D inputs against OHLC volatility estimators and alignment variants.")
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
        default=Path("results/transition_forecasting/modeling/cross_market_ohlc_transform_match"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, stage = load_stage_d(args.stage_d_run)
    panel = pd.read_csv(args.panel, usecols=["date", "open", "high", "low", "close", "ticker"])
    signals = build_ohlc_signals(panel)
    results = compare(manifest, stage, signals, max_samples=args.max_samples)
    results.to_csv(run_dir / "ohlc_transform_match_results.csv", index=False)
    top = results.head(20)
    top.to_csv(run_dir / "ohlc_transform_top20.csv", index=False)
    payload = {
        "test_evaluated": False,
        "candidates_evaluated": int(len(results)),
        "best": results.iloc[0].to_dict(),
        "top20": top.to_dict(orient="records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nTOP 20", flush=True)
    print(top.to_string(index=False), flush=True)
    print("\n" + json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
