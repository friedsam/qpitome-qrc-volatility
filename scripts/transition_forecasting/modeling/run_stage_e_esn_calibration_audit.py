from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from baselines.esn_representation import pool_trajectory, reservoir_trajectory, transform_input
from baselines.numpy_esn import make_esn_weights
from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import HAR_FEATURES, TARGET_COLUMNS, qlike_loss

ROLLING_SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_rolling_origin.py")
SPEC = importlib.util.spec_from_file_location("stage_e_rolling_origin", ROLLING_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROLLING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROLLING)

WINNER = {
    "representation": "level_diff_time",
    "pooling": "final_mean_std",
    "washout": 10,
    "n": 300,
    "sr": 0.9,
    "inp": 0.3,
    "leak": 0.3,
    "alpha": 1000.0,
}


def _scale_channels(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    train = x[mask].reshape(-1, x.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (x - mean[None, None, :]) / scale[None, None, :]


def _fit_har(manifest: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[mask]), y[mask])
    return model.predict(scaler.transform(x))


def _metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return (
        float(qlike_loss(y[mask], pred[mask]).mean()),
        float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))),
    )


def _chronological_calibration_split(manifest: pd.DataFrame, train_mask: np.ndarray, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    train = manifest.loc[train_mask, ["episode_id", "event_onset"]].copy()
    train["event_onset"] = pd.to_datetime(train["event_onset"], utc=True)
    episodes = train.groupby("episode_id", as_index=False)["event_onset"].min().sort_values("event_onset")
    n_cal = max(3, int(round(len(episodes) * fraction)))
    calibration_ids = set(episodes.tail(n_cal)["episode_id"])
    calibration = train_mask & manifest["episode_id"].isin(calibration_ids).to_numpy()
    fitting = train_mask & ~calibration
    return fitting, calibration


def _global_affine(y_cal: np.ndarray, pred_cal: np.ndarray, pred_all: np.ndarray) -> tuple[np.ndarray, float, float]:
    model = LinearRegression().fit(pred_cal.reshape(-1, 1), y_cal.reshape(-1))
    calibrated = model.predict(pred_all.reshape(-1, 1)).reshape(pred_all.shape)
    return calibrated, float(model.intercept_), float(model.coef_[0])


def _horizon_scale(y_cal: np.ndarray, pred_cal: np.ndarray, pred_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.sum(pred_cal * y_cal, axis=0)
    denominator = np.sum(pred_cal * pred_cal, axis=0)
    scale = numerator / np.maximum(denominator, 1e-12)
    return pred_all * scale[None, :], scale


def evaluate_fold(manifest: pd.DataFrame, sequences: np.ndarray, fold: int, seeds: tuple[int, ...]) -> pd.DataFrame:
    train_mask = manifest["fold_split"].eq("train").to_numpy()
    val_mask = manifest["fold_split"].eq("val").to_numpy()
    fit_mask, cal_mask = _chronological_calibration_split(manifest, train_mask)
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)

    har_fit = _fit_har(manifest, y, fit_mask)
    residual_fit = y - har_fit
    inputs = _scale_channels(transform_input(sequences, WINNER["representation"]), fit_mask)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        w_in, w = make_esn_weights(
            n_inputs=inputs.shape[-1],
            n_reservoir=WINNER["n"],
            spectral_radius=WINNER["sr"],
            input_scale=WINNER["inp"],
            seed=seed,
        )
        states = reservoir_trajectory(inputs, w_in, w, WINNER["leak"])
        features = pool_trajectory(states, WINNER["pooling"], WINNER["washout"])
        scaler = StandardScaler()
        model = Ridge(alpha=WINNER["alpha"])
        model.fit(scaler.fit_transform(features[fit_mask]), residual_fit[fit_mask])
        raw = har_fit + model.predict(scaler.transform(features))

        affine, intercept, slope = _global_affine(y[cal_mask], raw[cal_mask], raw)
        hscale, scales = _horizon_scale(y[cal_mask], raw[cal_mask], raw)

        for name, pred in (("raw", raw), ("global_affine", affine), ("horizon_scale", hscale)):
            qlike, rmse = _metrics(y, pred, val_mask)
            rows.append({
                "fold": fold,
                "seed": seed,
                "calibration": name,
                "val_qlike": qlike,
                "val_rmse": rmse,
                "fit_episodes": int(manifest.loc[fit_mask, "episode_id"].nunique()),
                "calibration_episodes": int(manifest.loc[cal_mask, "episode_id"].nunique()),
                "affine_intercept": intercept if name == "global_affine" else np.nan,
                "affine_slope": slope if name == "global_affine" else np.nan,
                "mean_horizon_scale": float(np.mean(scales)) if name == "horizon_scale" else np.nan,
            })
            print(f"fold={fold} seed={seed} {name:14s} qlike={qlike:.6f} rmse={rmse:.6f}", flush=True)
    return pd.DataFrame(rows)


def audit_cross_market_files(roots: tuple[Path, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    date_names = {"date", "datetime", "timestamp", "origin_date", "event_onset"}
    market_names = {"market", "market_group", "symbol", "ticker", "asset"}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".csv", ".parquet"} or not path.is_file():
                continue
            try:
                frame = pd.read_csv(path, nrows=5000) if path.suffix.lower() == ".csv" else pd.read_parquet(path).head(5000)
            except Exception as exc:
                rows.append({"path": str(path), "readable": False, "error": type(exc).__name__})
                continue
            lower = {str(c).lower(): str(c) for c in frame.columns}
            date_cols = [lower[name] for name in date_names if name in lower]
            market_cols = [lower[name] for name in market_names if name in lower]
            candidate = bool(date_cols and market_cols)
            row = {
                "path": str(path),
                "readable": True,
                "rows_sampled": int(len(frame)),
                "columns": int(len(frame.columns)),
                "date_columns": "|".join(date_cols),
                "market_columns": "|".join(market_cols),
                "cross_market_candidate": candidate,
            }
            if candidate:
                date_col, market_col = date_cols[0], market_cols[0]
                dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
                row.update({
                    "markets_sampled": int(frame[market_col].nunique(dropna=True)),
                    "dates_sampled": int(dates.nunique(dropna=True)),
                    "duplicate_market_dates": int(frame.assign(_date=dates).duplicated([market_col, "_date"]).sum()),
                    "min_date": str(dates.min()),
                    "max_date": str(dates.max()),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only ESN calibration diagnostic plus cross-market source audit.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--rolling-run", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--audit-roots", nargs="+", type=Path, default=[Path("data"), Path("results/transition_forecasting")])
    parser.add_argument("--out-dir", type=Path, default=Path("results/transition_forecasting/modeling/stage_e_esn_calibration_audit"))
    parser.add_argument("--run-id")
    args = parser.parse_args()

    _, sequences = ROLLING.load_stage_d(args.stage_d_run)
    assignments = pd.read_csv(args.rolling_run / "rolling_fold_manifest.csv")
    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)

    frames = []
    for fold in sorted(assignments["fold"].unique()):
        fold_manifest = assignments[assignments["fold"].eq(fold)].reset_index(drop=True)
        frames.append(evaluate_fold(fold_manifest, sequences, int(fold), tuple(args.seeds)))
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(run_dir / "calibration_results_by_seed.csv", index=False)
    summary = results.groupby("calibration", as_index=False).agg(
        mean_val_qlike=("val_qlike", "mean"),
        std_val_qlike=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
    ).sort_values("mean_val_qlike")
    summary.to_csv(run_dir / "calibration_summary.csv", index=False)

    audit = audit_cross_market_files(tuple(args.audit_roots))
    audit.to_csv(run_dir / "cross_market_file_audit.csv", index=False)
    candidates = audit[audit.get("cross_market_candidate", False).eq(True)] if not audit.empty else audit
    candidates.to_csv(run_dir / "cross_market_candidates.csv", index=False)

    payload = {
        "test_evaluated": False,
        "winner_configuration": WINNER,
        "calibration_protocol": "chronological final 20% of training episodes; validation untouched",
        "best_calibration": summary.iloc[0].to_dict(),
        "cross_market_candidate_files": int(len(candidates)),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nCALIBRATION SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nCross-market candidate files: {len(candidates)}", flush=True)
    if len(candidates):
        print(candidates[[c for c in ("path", "markets_sampled", "dates_sampled", "min_date", "max_date") if c in candidates]].to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
