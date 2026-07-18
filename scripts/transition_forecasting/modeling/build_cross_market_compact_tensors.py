from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.cross_market_mapping import MARKET_GROUP_TO_TICKER

CHANNELS = [
    "own_level",
    "own_diff",
    "global_median",
    "global_mad",
    "fraction_rising",
    "mean_abs_change",
    "directional_agreement",
    "availability_fraction",
    "time",
]


def load_panel(path: Path) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    frame = pd.read_csv(path, usecols=["date", "high", "low", "ticker"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    for column in ("high", "low"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(lower=1e-12)
    frame["signal"] = np.sqrt(np.log(frame["high"] / frame["low"]).clip(lower=0.0))
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    series = {
        str(ticker): group.set_index("date")["signal"].sort_index()
        for ticker, group in frame.groupby("ticker", sort=True)
    }
    wide = frame.pivot(index="date", columns="ticker", values="signal").sort_index()
    return series, wide


def aggregate_panel(wide: pd.DataFrame) -> pd.DataFrame:
    diff = wide.diff()
    median = wide.median(axis=1, skipna=True)
    mad = wide.sub(median, axis=0).abs().median(axis=1, skipna=True)
    rising = diff.gt(0).sum(axis=1) / diff.notna().sum(axis=1).replace(0, np.nan)
    mean_abs = diff.abs().mean(axis=1, skipna=True)
    pos = diff.gt(0).sum(axis=1)
    neg = diff.lt(0).sum(axis=1)
    valid = diff.notna().sum(axis=1).replace(0, np.nan)
    agreement = np.maximum(pos, neg) / valid
    availability = wide.notna().sum(axis=1) / float(wide.shape[1])
    return pd.DataFrame({
        "global_median": median,
        "global_mad": mad,
        "fraction_rising": rising,
        "mean_abs_change": mean_abs,
        "directional_agreement": agreement,
        "availability_fraction": availability,
    })


def build_raw_tensor(
    manifest: pd.DataFrame,
    series: dict[str, pd.Series],
    aggregates: pd.DataFrame,
    length: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = np.full((len(manifest), length, len(CHANNELS)), np.nan, dtype=np.float32)
    valid = np.zeros(len(manifest), dtype=bool)
    for i, row in manifest.reset_index(drop=True).iterrows():
        ticker = MARKET_GROUP_TO_TICKER[str(row["market_group"])]
        origin = pd.to_datetime(row["origin_date"], utc=True).normalize()
        own = series[ticker].loc[:origin].dropna().tail(length)
        if len(own) != length:
            continue
        dates = own.index
        agg = aggregates.reindex(dates)
        own_values = own.to_numpy(dtype=float)
        tensor[i, :, 0] = own_values
        tensor[i, :, 1] = np.diff(own_values, prepend=own_values[0])
        tensor[i, :, 2:8] = agg.to_numpy(dtype=float)
        tensor[i, :, 8] = np.linspace(0.0, 1.0, length)
        valid[i] = np.isfinite(tensor[i]).all()
    return tensor, valid


def normalize_fold(
    raw: np.ndarray,
    manifest: pd.DataFrame,
    fold_rows: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    sample_to_index = {str(sample_id): i for i, sample_id in enumerate(manifest["sample_id"].astype(str))}
    train_ids = fold_rows.loc[fold_rows["fold_split"].eq("train"), "sample_id"].astype(str)
    train_idx = np.array([sample_to_index[s] for s in train_ids if s in sample_to_index], dtype=int)
    if len(train_idx) == 0:
        raise ValueError("fold has no train samples")
    out = raw.copy().astype(np.float32)

    market_stats: dict[str, dict[str, float]] = {}
    train_manifest = manifest.iloc[train_idx].reset_index(drop=True)
    for market_group, group in train_manifest.groupby("market_group"):
        local_idx = train_idx[group.index.to_numpy()]
        values = raw[local_idx, :, 0].reshape(-1)
        values = values[np.isfinite(values)]
        mean = float(values.mean())
        std = float(max(values.std(), 1e-8))
        market_stats[str(market_group)] = {"mean": mean, "std": std}
        all_idx = manifest.index[manifest["market_group"].astype(str).eq(str(market_group))].to_numpy()
        out[all_idx, :, 0] = (out[all_idx, :, 0] - mean) / std
        out[all_idx, :, 1] = out[all_idx, :, 1] / std

    global_stats: dict[str, dict[str, float]] = {}
    for channel in range(2, 7):
        values = raw[train_idx, :, channel].reshape(-1)
        values = values[np.isfinite(values)]
        mean = float(values.mean())
        std = float(max(values.std(), 1e-8))
        out[:, :, channel] = (out[:, :, channel] - mean) / std
        global_stats[CHANNELS[channel]] = {"mean": mean, "std": std}

    return out, {"market_stats": market_stats, "global_stats": global_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact cross-market tensors with train-only fold normalization.")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--rolling-manifest", type=Path, required=True)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/raw/transition_forecasting/global_stock_indices_historical_data/all_indices_data.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/cross_market_compact_tensors"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest = pd.read_csv(args.sample_manifest).reset_index(drop=True)
    rolling = pd.read_csv(args.rolling_manifest)
    series, wide = load_panel(args.panel)
    aggregates = aggregate_panel(wide)
    raw, valid = build_raw_tensor(manifest, series, aggregates)

    manifest_out = manifest.copy()
    manifest_out["cross_market_valid"] = valid
    manifest_out.to_csv(run_dir / "sample_manifest.csv", index=False)
    np.savez_compressed(
        run_dir / "raw_compact_tensor.npz",
        X=raw,
        sample_id=manifest["sample_id"].astype(str).to_numpy(),
        channel_names=np.asarray(CHANNELS),
        valid=valid,
    )

    fold_summary = []
    for fold in sorted(rolling["fold"].unique()):
        fold_rows = rolling.loc[rolling["fold"].eq(fold)].copy()
        normalized, stats = normalize_fold(raw, manifest, fold_rows)
        fold_dir = run_dir / f"fold_{int(fold)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "compact_tensor.npz",
            X=normalized,
            sample_id=manifest["sample_id"].astype(str).to_numpy(),
            channel_names=np.asarray(CHANNELS),
            valid=valid,
        )
        (fold_dir / "normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
        split_counts = fold_rows.groupby("fold_split")["sample_id"].nunique().to_dict()
        fold_summary.append({"fold": int(fold), **{str(k): int(v) for k, v in split_counts.items()}})
        print(f"fold={int(fold)} valid={int(valid.sum())}/{len(valid)} splits={split_counts}", flush=True)

    summary = {
        "test_evaluated": False,
        "shape": list(raw.shape),
        "channels": CHANNELS,
        "valid_samples": int(valid.sum()),
        "invalid_samples": int((~valid).sum()),
        "folds": fold_summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
