"""Long-history replication of the branch-conditioned HAR residual quickshot.

Purpose
-------
Test whether the tentative relapse-only residual signal survives on the 1950--2026
GSPC history. This is a replication, not a new search.

Locked choices from the modern-sample exploration:
- target: log(actual future RV20 / causal HAR prediction)
- lookback: 40 rows
- representations: full path linear and reset ESN only
- ESN: 50 units, rho=.9, input scale=.3, leak=.3, seeds 0/1/2
- Ridge readout alphas: 1, 10, 100, 1000
- input families: generic path and volatility-failure
- prequential protocol: min_train=18, outcome horizon=40, embargo=0

Important baseline qualification
--------------------------------
The pre-1993 history has no VIX. Therefore this script uses a locked VIX-blind
HAR-RV model with [RV5, RV10, RV20, RV60], StandardScaler, Ridge(alpha=1), and
raw future RV20 target. It is not called the exact canonical HAR.

The script auto-detects the local long-history daily CSV and long-history episode
CSV. Explicit --data and --episodes arguments override auto-detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.numpy_esn import esn_states, make_esn_weights
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)

LOOKBACK = 40
SEEDS = (0, 1, 2)
ESN_CONFIG = dict(n_reservoir=50, spectral_radius=0.9, input_scale=0.3, leak=0.3)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
HAR_FEATURES = ("rv_5d", "rv_10d", "rv_20d", "rv_60d")
HAR_TARGET = "future_rv_20d"
HAR_TARGET_HORIZON = 20
HAR_MIN_TRAIN = 504
DEFAULT_OUTPUT = Path("results/baselines/long_history_har_residual_replication_v1")

GENERIC_COLUMNS = (
    "market_log_return",
    "market_abs_log_return",
    "log_rv5_over_rv20",
    "log_rv20_over_rv60",
    "drawdown_120d_path",
    "rv5_change_scaled",
)

VOLATILITY_FAILURE_COLUMNS = (
    "log_rv5_over_rv20",
    "log_rv20_over_rv60",
    "d_log_rv5_over_rv20_5d",
    "d_log_rv20_over_rv60_5d",
    "rv5_acceleration_scaled",
    "rv20_change_scaled",
    "rv5_distance_from_20d_peak",
    "rv5_reexpansion_count_10d",
    "rv5_reexpansion_magnitude_10d",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--episodes", type=Path)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _date_column(df: pd.DataFrame) -> str | None:
    for name in ("date", "Date", "DATE", "branch_date"):
        if name in df.columns:
            return name
    return None


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col = _date_column(out)
    if col is None:
        raise KeyError("No date column found")
    if col != "date":
        out = out.rename(columns={col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _candidate_csvs() -> list[Path]:
    roots = [Path("data"), Path("results")]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob("*.csv"))
    return sorted(set(paths))


def detect_long_daily() -> Path:
    candidates: list[tuple[tuple[int, int, int], Path, str]] = []
    for path in _candidate_csvs():
        try:
            df = _read_csv(path)
            col = _date_column(df)
            if col is None or len(df) < 12000:
                continue
            dates = pd.to_datetime(df[col], errors="coerce")
            if dates.notna().sum() < 12000:
                continue
            start = dates.min()
            end = dates.max()
            if start.year > 1965 or end.year < 2024:
                continue
            rv_count = sum(c in df.columns for c in (*HAR_FEATURES, HAR_TARGET))
            score = (rv_count, len(df), -start.year)
            candidates.append((score, path, f"{start.date()}..{end.date()}, n={len(df)}, rv_cols={rv_count}"))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect a 1950s--2020s daily CSV under data/ or results/. "
            "Pass --data PATH."
        )
    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best[0]:
        detail = "\n".join(f"  {p}: {d}" for _, p, d in candidates[:10])
        raise RuntimeError(f"Ambiguous long-history daily CSV candidates:\n{detail}\nPass --data PATH.")
    print(f"Auto-detected long daily file: {best[1]} ({best[2]})")
    return best[1]


def detect_long_episodes() -> Path:
    candidates: list[tuple[tuple[int, int, int], Path, str]] = []
    required = {"episode_id", "outcome"}
    for path in _candidate_csvs():
        try:
            df = _read_csv(path)
            if not required.issubset(df.columns) or not (60 <= len(df) <= 200):
                continue
            has_branch_idx = "branch_idx" in df.columns
            date_col = "branch_date" if "branch_date" in df.columns else _date_column(df)
            if date_col is None:
                continue
            dates = pd.to_datetime(df[date_col], errors="coerce")
            if dates.notna().sum() < 60 or dates.min().year > 1970 or dates.max().year < 2020:
                continue
            valid_outcomes = df["outcome"].astype(str).str.lower().isin({"recovery", "relapse", "mixed"}).mean()
            if valid_outcomes < 0.9:
                continue
            score = (int(has_branch_idx), len(df), -dates.min().year)
            candidates.append((score, path, f"{dates.min().date()}..{dates.max().date()}, n={len(df)}"))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect the long-history episode CSV under data/ or results/. "
            "Pass --episodes PATH."
        )
    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best[0]:
        detail = "\n".join(f"  {p}: {d}" for _, p, d in candidates[:10])
        raise RuntimeError(f"Ambiguous long-history episode CSV candidates:\n{detail}\nPass --episodes PATH.")
    print(f"Auto-detected long episode file: {best[1]} ({best[2]})")
    return best[1]


def normalize_daily(raw: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_date(raw).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    aliases = {
        "rv_5d": ("rv_5d", "rv5", "RV5", "rv5d"),
        "rv_10d": ("rv_10d", "rv10", "RV10", "rv10d"),
        "rv_20d": ("rv_20d", "rv20", "RV20", "rv20d"),
        "rv_60d": ("rv_60d", "rv60", "RV60", "rv60d"),
        "future_rv_20d": ("future_rv_20d", "future_rv20", "rv20_future", "future_rv20d"),
    }
    for canonical, names in aliases.items():
        if canonical not in out.columns:
            found = next((n for n in names if n in out.columns), None)
            if found is not None:
                out = out.rename(columns={found: canonical})

    price_col = next(
        (c for c in ("adj_close", "Adj Close", "adjusted_close", "close", "Close", "gspc_adj_close", "spy_adj_close") if c in out.columns),
        None,
    )
    return_col = next(
        (c for c in ("log_return", "market_log_return", "gspc_log_return", "spy_log_return") if c in out.columns),
        None,
    )
    if return_col is not None:
        out["market_log_return"] = pd.to_numeric(out[return_col], errors="coerce")
    elif price_col is not None:
        price = pd.to_numeric(out[price_col], errors="coerce")
        out["market_log_return"] = np.log(price).diff()
    else:
        raise KeyError("Need a price or log-return column in the long daily file")

    if price_col is None:
        # Reconstruct a relative price path solely for drawdown geometry.
        out["market_price"] = np.exp(out["market_log_return"].fillna(0.0).cumsum())
    else:
        out["market_price"] = pd.to_numeric(out[price_col], errors="coerce")

    missing = [c for c in (*HAR_FEATURES, HAR_TARGET) if c not in out.columns]
    if missing:
        raise KeyError(
            f"Long daily file is missing required realized-volatility columns: {missing}. "
            "Use the engineered long-history file, not raw GSPC prices."
        )
    for col in (*HAR_FEATURES, HAR_TARGET):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_episodes(raw: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    if "branch_date" not in out.columns:
        col = _date_column(out)
        if col is None:
            raise KeyError("Episode table needs branch_date/date")
        out = out.rename(columns={col: "branch_date"})
    out["branch_date"] = pd.to_datetime(out["branch_date"], errors="coerce")
    out["outcome"] = out["outcome"].astype(str).str.lower()
    out = out[out["outcome"].isin(["recovery", "relapse", "mixed"])].copy()
    out = out.sort_values("branch_date").reset_index(drop=True)

    date_to_idx = pd.Series(daily.index.to_numpy(), index=daily["date"]).to_dict()
    mapped = out["branch_date"].map(date_to_idx)
    if mapped.isna().any():
        missing = out.loc[mapped.isna(), "branch_date"].dt.strftime("%Y-%m-%d").tolist()[:10]
        raise ValueError(f"Episode branch dates absent from long daily file: {missing}")
    if "branch_idx" in out.columns:
        supplied = pd.to_numeric(out["branch_idx"], errors="coerce")
        mismatch = supplied.notna() & (supplied.astype("Int64") != mapped.astype("Int64"))
        if mismatch.any():
            print(f"Remapping {int(mismatch.sum())} branch_idx values to the selected long daily file.")
    out["branch_idx"] = mapped.astype(int)
    if "episode_id" not in out.columns:
        out["episode_id"] = np.arange(1, len(out) + 1)
    out["episode_id"] = pd.to_numeric(out["episode_id"], errors="raise").astype(int)
    return out


def add_generic_channels(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["market_abs_log_return"] = out["market_log_return"].abs()
    out["log_rv5_over_rv20"] = np.log(out["rv_5d"] / out["rv_20d"])
    out["log_rv20_over_rv60"] = np.log(out["rv_20d"] / out["rv_60d"])
    peak = out["market_price"].rolling(120).max()
    out["drawdown_120d_path"] = out["market_price"] / peak - 1.0
    out["rv5_change_scaled"] = out["rv_5d"].diff(5) / out["rv_20d"]
    bounds = {
        "market_log_return": (-0.15, 0.15),
        "market_abs_log_return": (0.0, 0.15),
        "log_rv5_over_rv20": (-2.0, 2.0),
        "log_rv20_over_rv60": (-1.5, 1.5),
        "drawdown_120d_path": (-0.60, 0.05),
        "rv5_change_scaled": (-3.0, 3.0),
    }
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].clip(lo, hi)
    return out


def add_volatility_failure_channels(daily: pd.DataFrame) -> pd.DataFrame:
    out = add_generic_channels(daily)
    out["d_log_rv5_over_rv20_5d"] = out["log_rv5_over_rv20"].diff(5)
    out["d_log_rv20_over_rv60_5d"] = out["log_rv20_over_rv60"].diff(5)
    rv5_change = out["rv_5d"].diff(5)
    out["rv5_acceleration_scaled"] = (rv5_change - rv5_change.shift(5)) / out["rv_20d"]
    out["rv20_change_scaled"] = out["rv_20d"].diff(5) / out["rv_20d"]
    out["rv5_distance_from_20d_peak"] = np.log(out["rv_5d"] / out["rv_5d"].rolling(20).max())
    up = (out["rv_5d"].diff() > 0).astype(float)
    pos = out["rv_5d"].diff().clip(lower=0.0)
    out["rv5_reexpansion_count_10d"] = up.rolling(10).sum() / 10.0
    out["rv5_reexpansion_magnitude_10d"] = pos.rolling(10).sum() / out["rv_20d"]
    bounds = {
        "d_log_rv5_over_rv20_5d": (-2.0, 2.0),
        "d_log_rv20_over_rv60_5d": (-1.5, 1.5),
        "rv5_acceleration_scaled": (-4.0, 4.0),
        "rv20_change_scaled": (-2.0, 2.0),
        "rv5_distance_from_20d_peak": (-3.0, 0.1),
        "rv5_reexpansion_count_10d": (0.0, 1.0),
        "rv5_reexpansion_magnitude_10d": (0.0, 4.0),
    }
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].clip(lo, hi)
    return out


def causal_har_predictions(daily: pd.DataFrame, episodes: pd.DataFrame) -> np.ndarray:
    x_all = daily.loc[:, HAR_FEATURES].to_numpy(float)
    y_all = daily[HAR_TARGET].to_numpy(float)
    preds: list[float] = []
    for row in episodes.itertuples(index=False):
        branch_idx = int(row.branch_idx)
        latest = branch_idx - HAR_TARGET_HORIZON - 1
        if latest < 0:
            raise ValueError(f"No leakage-safe HAR rows for episode {row.episode_id}")
        x = x_all[: latest + 1]
        y = y_all[: latest + 1]
        finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
        x_train = x[finite]
        y_train = y[finite]
        if len(y_train) < HAR_MIN_TRAIN:
            raise ValueError(f"Episode {row.episode_id}: only {len(y_train)} HAR training rows")
        test = x_all[[branch_idx]]
        if not np.isfinite(test).all():
            raise ValueError(f"Non-finite HAR features at episode {row.episode_id}")
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_train, y_train)
        preds.append(max(float(model.predict(test)[0]), 1e-8))
    return np.asarray(preds)


def extract_windows(daily: pd.DataFrame, episodes: pd.DataFrame, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = daily.loc[:, columns].to_numpy(float)
    ids: list[int] = []
    windows: list[np.ndarray] = []
    for row in episodes.itertuples(index=False):
        end = int(row.branch_idx) + 1
        start = end - LOOKBACK
        window = values[start:end]
        if len(window) != LOOKBACK or not np.isfinite(window).all():
            raise ValueError(f"Incomplete/non-finite {columns[0]} path for episode {row.episode_id}")
        ids.append(int(row.episode_id))
        windows.append(window)
    return np.asarray(ids), np.asarray(windows)


def build_feature_maps(daily: pd.DataFrame, episodes: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict[int, np.ndarray]]:
    ids, windows = extract_windows(daily, episodes, columns)
    maps: dict[str, dict[int, np.ndarray]] = {
        "full_path_linear": {int(i): windows[k].reshape(-1) for k, i in enumerate(ids)}
    }
    for seed in SEEDS:
        w_in, w = make_esn_weights(
            n_inputs=windows.shape[2],
            n_reservoir=ESN_CONFIG["n_reservoir"],
            spectral_radius=ESN_CONFIG["spectral_radius"],
            input_scale=ESN_CONFIG["input_scale"],
            seed=seed,
        )
        feats = esn_states(windows, w_in, w, ESN_CONFIG["leak"])
        maps[f"reset_esn_seed{seed}"] = {int(i): feats[k] for k, i in enumerate(ids)}
    return maps


def fit_residual(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[float, float]:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])
    model.fit(x_train, y_train)
    return float(model.predict(x_test)[0]), float(model.named_steps["ridge"].alpha_)


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.maximum(np.asarray(y, float), 1e-12)
    yhat = np.maximum(np.asarray(yhat, float), 1e-12)
    ratio = y / yhat
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mz_fit(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(yhat)), yhat])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(coef[0]), float(coef[1])


def metric_row(family: str, model: str, g: pd.DataFrame) -> dict[str, object]:
    y = g["actual_future_rv20"].to_numpy(float)
    yhat = g["combined_future_rv20_pred"].to_numpy(float)
    r = g["har_log_residual_actual"].to_numpy(float)
    rhat = g["har_log_residual_pred"].to_numpy(float)
    sse = float(np.sum((r - rhat) ** 2))
    sse0 = float(np.sum(r**2))
    intercept, slope = mz_fit(y, yhat)
    return {
        "family": family,
        "model": model,
        "n_oos": len(g),
        "future_rv_rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
        "future_rv_mae": float(np.mean(np.abs(yhat - y))),
        "qlike": qlike(y, yhat),
        "mz_intercept": intercept,
        "mz_slope": slope,
        "har_residual_rmse": float(np.sqrt(np.mean((rhat - r) ** 2))),
        "residual_r2_vs_har": 1.0 - sse / sse0 if sse0 > 0 else np.nan,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_path = args.data or detect_long_daily()
    episode_path = args.episodes or detect_long_episodes()

    daily = normalize_daily(_read_csv(data_path))
    episodes = normalize_episodes(_read_csv(episode_path), daily)
    print(f"Long daily rows: {len(daily)}, {daily['date'].min().date()}..{daily['date'].max().date()}")
    print(f"Long episodes: {len(episodes)}, {episodes['branch_date'].min().date()}..{episodes['branch_date'].max().date()}")
    print("Outcome counts:")
    print(episodes["outcome"].value_counts().to_string())

    episodes["har_future_rv20_pred"] = causal_har_predictions(daily, episodes)
    episodes["actual_future_rv20"] = daily.iloc[episodes["branch_idx"].to_numpy()][HAR_TARGET].to_numpy(float)
    episodes["har_log_residual_actual"] = np.log(
        episodes["actual_future_rv20"] / episodes["har_future_rv20_pred"]
    )
    if not np.isfinite(episodes["har_log_residual_actual"]).all():
        raise ValueError("Non-finite long-history HAR residual targets")

    family_frames = {
        "generic": (add_generic_channels(daily), GENERIC_COLUMNS),
        "volatility_failure": (add_volatility_failure_channels(daily), VOLATILITY_FAILURE_COLUMNS),
    }
    feature_maps = {
        family: build_feature_maps(frame, episodes, columns)
        for family, (frame, columns) in family_frames.items()
    }

    protocol = EpisodePrequentialConfig(min_train_episodes=18, outcome_horizon_rows=40, embargo_rows=0)
    assignments, steps = make_episode_prequential_steps(episodes, protocol)
    lookup = episodes.set_index("episode_id", drop=False)
    rows: list[dict[str, object]] = []

    for step in steps.itertuples(index=False):
        test_id = int(step.test_episode_id)
        test = lookup.loc[test_id]
        train_ids = assignments[
            (assignments["step"] == step.step) & (assignments["split"] == "train")
        ]["episode_id"].astype(int).tolist()
        y_train = lookup.loc[train_ids, "har_log_residual_actual"].to_numpy(float)

        for family, model_maps in feature_maps.items():
            rows.append({
                "family": family,
                "model": "har_zero_residual",
                "step": int(step.step),
                "episode_id": test_id,
                "branch_date": test["branch_date"],
                "outcome": test["outcome"],
                "n_train": len(train_ids),
                "selected_alpha": np.nan,
                "har_log_residual_actual": float(test["har_log_residual_actual"]),
                "har_log_residual_pred": 0.0,
                "har_future_rv20_pred": float(test["har_future_rv20_pred"]),
                "actual_future_rv20": float(test["actual_future_rv20"]),
            })
            for model_name, fmap in model_maps.items():
                pred, alpha = fit_residual(
                    np.vstack([fmap[i] for i in train_ids]),
                    y_train,
                    fmap[test_id][None, :],
                )
                rows.append({
                    "family": family,
                    "model": model_name,
                    "step": int(step.step),
                    "episode_id": test_id,
                    "branch_date": test["branch_date"],
                    "outcome": test["outcome"],
                    "n_train": len(train_ids),
                    "selected_alpha": alpha,
                    "har_log_residual_actual": float(test["har_log_residual_actual"]),
                    "har_log_residual_pred": pred,
                    "har_future_rv20_pred": float(test["har_future_rv20_pred"]),
                    "actual_future_rv20": float(test["actual_future_rv20"]),
                })

    pred = pd.DataFrame(rows)
    pred["combined_future_rv20_pred"] = (
        pred["har_future_rv20_pred"] * np.exp(pred["har_log_residual_pred"])
    ).clip(lower=1e-8)

    ensemble_rows = []
    for family in family_frames:
        family_pred = pred[pred["family"] == family]
        seed_models = [f"reset_esn_seed{s}" for s in SEEDS]
        subset = family_pred[family_pred["model"].isin(seed_models)]
        wide = subset.pivot(index="episode_id", columns="model", values="har_log_residual_pred")
        ensemble = wide.mean(axis=1)
        base = subset[subset["model"] == seed_models[0]].copy()
        base["model"] = "reset_esn_ensemble"
        base["selected_alpha"] = np.nan
        base["har_log_residual_pred"] = base["episode_id"].map(ensemble)
        base["combined_future_rv20_pred"] = (
            base["har_future_rv20_pred"] * np.exp(base["har_log_residual_pred"])
        ).clip(lower=1e-8)
        ensemble_rows.append(base)
    pred = pd.concat([pred, *ensemble_rows], ignore_index=True)

    summary = pd.DataFrame([
        metric_row(family, model, g)
        for (family, model), g in pred.groupby(["family", "model"])
    ]).sort_values(["future_rv_rmse", "qlike"])

    outcome_rows = []
    for (family, model, outcome), g in pred.groupby(["family", "model", "outcome"]):
        row = metric_row(family, model, g)
        row["outcome"] = outcome
        outcome_rows.append(row)
    by_outcome = pd.DataFrame(outcome_rows)

    pred.to_csv(args.output / "oos_predictions.csv", index=False)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)
    by_outcome.to_csv(args.output / "metrics_by_outcome.csv", index=False)
    episodes.to_csv(args.output / "long_episode_targets.csv", index=False)
    pd.DataFrame([
        {"family": family, "channel": channel}
        for family, (_, columns) in family_frames.items()
        for channel in columns
    ]).to_csv(args.output / "input_families.csv", index=False)
    (args.output / "run_manifest.json").write_text(json.dumps({
        "status": "locked long-history replication",
        "daily_file": str(data_path),
        "episode_file": str(episode_path),
        "baseline": "VIX-blind HAR-RV: RV5/RV10/RV20/RV60 -> StandardScaler -> Ridge(alpha=1)",
        "baseline_qualification": "not exact canonical HAR because pre-1993 VIX is unavailable",
        "target": "log(actual future RV20 / causal VIX-blind HAR-RV prediction)",
        "families": {"generic": list(GENERIC_COLUMNS), "volatility_failure": list(VOLATILITY_FAILURE_COLUMNS)},
        "lookback": LOOKBACK,
        "esn_config": ESN_CONFIG,
        "seeds": list(SEEDS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "protocol": protocol.__dict__,
    }, indent=2) + "\n")

    print("\nSummary metrics:")
    print(summary.to_string(index=False))
    print("\nOutcome metrics for HAR, linear path, and ESN ensemble:")
    keep = by_outcome[by_outcome["model"].isin(["har_zero_residual", "full_path_linear", "reset_esn_ensemble"])]
    print(keep.sort_values(["family", "outcome", "future_rv_rmse"]).to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
