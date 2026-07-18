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


def build_range_formulas(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for ticker, group in panel.groupby("ticker", sort=True):
        frame = group[["date", "high", "low", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.normalize()
        frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
        for column in ("high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(lower=1e-12)

        high = frame["high"]
        low = frame["low"]
        close = frame["close"]
        log_range = np.log(high / low)
        range_over_low = (high - low) / low
        range_over_close = (high - low) / close
        range_over_mid = (high - low) / ((high + low) / 2.0)

        formulas = {
            "log_high_low": log_range,
            "pct_log_high_low": 100.0 * log_range,
            "range_over_low": range_over_low,
            "pct_range_over_low": 100.0 * range_over_low,
            "range_over_close": range_over_close,
            "pct_range_over_close": 100.0 * range_over_close,
            "range_over_mid": range_over_mid,
            "pct_range_over_mid": 100.0 * range_over_mid,
            "sqrt_log_high_low": np.sqrt(log_range.clip(lower=0.0)),
            "squared_log_high_low": log_range.pow(2),
            "parkinson_variance": log_range.pow(2) / (4.0 * np.log(2.0)),
            "parkinson_sigma": np.sqrt(log_range.pow(2) / (4.0 * np.log(2.0))),
            "annualized_parkinson_sigma": np.sqrt(252.0) * np.sqrt(log_range.pow(2) / (4.0 * np.log(2.0))),
        }
        output[str(ticker)] = pd.DataFrame(formulas, index=frame.index)
    return output


def collect_pairs(
    manifest: pd.DataFrame,
    stage: np.ndarray,
    signals: dict[str, pd.DataFrame],
    candidate: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for index, row in manifest.iterrows():
        market_group = str(row["market_group"])
        ticker = MARKET_GROUP_TO_TICKER[market_group]
        origin = pd.to_datetime(row["origin_date"], utc=True).normalize()
        series = signals[ticker][candidate].loc[:origin].dropna().tail(40)
        if len(series) != 40:
            continue
        rows.append(pd.DataFrame({
            "market_group": market_group,
            "stage": stage[index],
            "raw": series.to_numpy(dtype=float),
        }))
    if not rows:
        return pd.DataFrame(columns=["market_group", "stage", "raw"])
    return pd.concat(rows, ignore_index=True)


def affine_metrics(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.std(x) <= 0.0:
        raise ValueError("candidate has fewer than three finite varying paired observations")
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    return {
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean((y - fitted) ** 2))),
        "mae": float(np.mean(np.abs(y - fitted))),
    }


def winsorize(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("cannot winsorize an empty candidate")
    lo, hi = np.quantile(finite, [lower, upper])
    return np.clip(values, lo, hi)


def evaluate_candidate(pairs: pd.DataFrame, candidate: str) -> list[dict[str, object]]:
    if pairs.empty:
        raise ValueError(f"candidate {candidate!r} produced no aligned 40-day windows")
    y = pairs["stage"].to_numpy(dtype=float)
    x = pairs["raw"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    base = affine_metrics(y, x)
    rows.append({"candidate": candidate, "normalization": "global_affine", **base})

    for lower, upper in ((0.001, 0.999), (0.005, 0.995), (0.01, 0.99)):
        clipped = winsorize(x, lower, upper)
        rows.append({
            "candidate": candidate,
            "normalization": f"winsor_{lower:g}_{upper:g}",
            **affine_metrics(y, clipped),
        })

    market_fitted = np.empty_like(y)
    market_z_raw = np.empty_like(x)
    market_z_stage = np.empty_like(y)
    for _, group in pairs.groupby("market_group"):
        idx = group.index.to_numpy()
        xg = group["raw"].to_numpy(dtype=float)
        yg = group["stage"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(xg, yg, deg=1)
        market_fitted[idx] = intercept + slope * xg
        market_z_raw[idx] = (xg - xg.mean()) / max(xg.std(), 1e-12)
        market_z_stage[idx] = (yg - yg.mean()) / max(yg.std(), 1e-12)

    rows.append({
        "candidate": candidate,
        "normalization": "per_market_affine",
        "correlation": float(np.corrcoef(y, market_fitted)[0, 1]),
        "slope": np.nan,
        "intercept": np.nan,
        "rmse": float(np.sqrt(np.mean((y - market_fitted) ** 2))),
        "mae": float(np.mean(np.abs(y - market_fitted))),
    })
    rows.append({
        "candidate": candidate,
        "normalization": "per_market_zscore",
        "correlation": float(np.corrcoef(market_z_stage, market_z_raw)[0, 1]),
        "slope": np.nan,
        "intercept": np.nan,
        "rmse": float(np.sqrt(np.mean((market_z_stage - market_z_raw) ** 2))),
        "mae": float(np.mean(np.abs(market_z_stage - market_z_raw))),
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused audit of high-low range formulas and market-specific normalization.")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/raw/transition_forecasting/global_stock_indices_historical_data/all_indices_data.csv"),
    )
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/cross_market_range_formula_audit"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest, stage = load_stage_d(args.stage_d_run)
    panel = pd.read_csv(args.panel, usecols=["date", "high", "low", "close", "ticker"])
    signals = build_range_formulas(panel)

    rows: list[dict[str, object]] = []
    for candidate in next(iter(signals.values())).columns:
        pairs = collect_pairs(manifest, stage, signals, candidate)
        candidate_rows = evaluate_candidate(pairs, candidate)
        rows.extend(candidate_rows)
        best = min(candidate_rows, key=lambda row: float(row["rmse"]))
        print(
            f"{candidate:30s} pairs={len(pairs):7d} best={best['normalization']:20s} "
            f"corr={float(best['correlation']):.6f} rmse={float(best['rmse']):.6f}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values(["rmse", "correlation"], ascending=[True, False])
    results.to_csv(run_dir / "range_formula_results.csv", index=False)
    top = results.head(25)
    top.to_csv(run_dir / "range_formula_top25.csv", index=False)
    payload = {
        "test_evaluated": False,
        "formulas": int(results["candidate"].nunique()),
        "normalizations": sorted(results["normalization"].unique()),
        "best": top.iloc[0].to_dict(),
        "top25": top.to_dict(orient="records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nTOP 25", flush=True)
    print(top.to_string(index=False), flush=True)
    print("\n" + json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
