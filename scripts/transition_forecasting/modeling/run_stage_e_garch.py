from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.garch import (
    GARCHConfig,
    config_to_dict,
    fit_garch_variance_path,
    variance_path_to_log_volatility_path,
)
from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    TARGET_COLUMNS,
    _metric_rows,
)


def _latest_complete_run(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "sample_manifest.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no complete Stage D run found under {root}")
    return candidates[-1]


def _load_close_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    normalized = {column.strip().lower(): column for column in frame.columns}
    required = {"date", "close"}
    missing = required.difference(normalized)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[normalized["date"]], errors="coerce"),
            "close": pd.to_numeric(frame[normalized["close"]], errors="coerce"),
        }
    ).dropna()
    result = result[result["close"] > 0]
    result = result.drop_duplicates("date", keep="last").sort_values("date")
    return result.set_index("date")["close"].astype(float)


def _return_history(close: pd.Series, origin_date: pd.Timestamp, history: int) -> np.ndarray:
    available = close.loc[:origin_date]
    returns = np.log(available).diff().dropna()
    if history > 0:
        returns = returns.iloc[-history:]
    return returns.to_numpy(dtype=float)


def run_stage_e_garch(
    manifest: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    history: int = 2500,
    minimum_history: int = 250,
    config: GARCHConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cfg = config or GARCHConfig()
    required_manifest = {"sample_id", "episode_id", "index", "origin_date", "label", "lead", "split", *TARGET_COLUMNS}
    missing_manifest = required_manifest.difference(manifest.columns)
    if missing_manifest:
        raise ValueError(f"manifest missing columns: {sorted(missing_manifest)}")
    required_inventory = {"index", "path"}
    missing_inventory = required_inventory.difference(inventory.columns)
    if missing_inventory:
        raise ValueError(f"inventory missing columns: {sorted(missing_inventory)}")

    validation = manifest[manifest["split"].eq("val")].copy().reset_index(drop=True)
    validation["origin_date"] = pd.to_datetime(validation["origin_date"])
    path_by_index = {
        str(row["index"]): Path(str(row["path"])) for _, row in inventory.iterrows()
    }
    closes = {
        index_name: _load_close_series(path_by_index[index_name])
        for index_name in sorted(validation["index"].astype(str).unique())
    }

    predictions: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    for _, row in validation.iterrows():
        index_name = str(row["index"])
        returns = _return_history(closes[index_name], pd.Timestamp(row["origin_date"]), history)
        if len(returns) < minimum_history:
            prediction = np.full(len(TARGET_COLUMNS), np.nan)
            diagnostics.append(
                {
                    "sample_id": row["sample_id"],
                    "index": index_name,
                    "history_rows": int(len(returns)),
                    "converged": False,
                    "convergence_flag": None,
                    "fit_note": f"insufficient_history:{len(returns)}<{minimum_history}",
                }
            )
        else:
            forecast = fit_garch_variance_path(
                returns,
                horizon=len(TARGET_COLUMNS),
                config=cfg,
            )
            prediction = (
                variance_path_to_log_volatility_path(
                    forecast.variance_path,
                    return_scale=cfg.return_scale,
                )
                if forecast.converged
                else np.full(len(TARGET_COLUMNS), np.nan)
            )
            diagnostics.append(
                {
                    "sample_id": row["sample_id"],
                    "index": index_name,
                    "history_rows": int(len(returns)),
                    "converged": bool(forecast.converged),
                    "convergence_flag": forecast.convergence_flag,
                    "fit_note": forecast.note,
                }
            )
        predictions.append(prediction)

    y_true = validation[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    finite_rows = np.all(np.isfinite(y_pred), axis=1)

    output = validation[["sample_id", "episode_id", "index", "origin_date", "label", "lead", "split"]].copy()
    output.insert(0, "model", "garch_1_1_t")
    for h in range(len(TARGET_COLUMNS)):
        output[f"actual_h{h + 1}"] = y_true[:, h]
        output[f"predicted_h{h + 1}"] = y_pred[:, h]
    diagnostics_frame = pd.DataFrame(diagnostics)
    output = output.merge(diagnostics_frame, on=["sample_id", "index"], how="left", validate="one_to_one")

    metric_rows = _metric_rows(
        "garch_1_1_t",
        validation.loc[finite_rows].reset_index(drop=True),
        y_true[finite_rows],
        y_pred[finite_rows],
    ) if finite_rows.any() else []
    metrics = pd.DataFrame(metric_rows)
    summary = {
        "selection_split": "val",
        "test_evaluated": False,
        "model": "garch_1_1_t",
        "target_contract": "daily log volatility path; no annualization or path aggregation",
        "history": history,
        "minimum_history": minimum_history,
        "garch_config": config_to_dict(cfg),
        "n_validation_samples": int(len(validation)),
        "n_converged": int(finite_rows.sum()),
        "n_failed": int((~finite_rows).sum()),
        "convergence_rate": float(finite_rows.mean()) if len(finite_rows) else None,
        "markets": int(validation["index"].nunique()),
    }
    return output, metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the existing GARCH(1,1)-t baseline on the Stage E validation samples."
    )
    parser.add_argument(
        "--stage-d-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_d_dataset"),
    )
    parser.add_argument("--stage-d-run", type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "results/transition_forecasting/quality/global_index_ohlc_audit/"
            "global_index_audit_001/global_index_ohlc_inventory.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_garch"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--history", type=int, default=2500)
    parser.add_argument("--minimum-history", type=int, default=250)
    args = parser.parse_args()

    stage_d_run = args.stage_d_run or _latest_complete_run(args.stage_d_root)
    resolved = vars(args).copy()
    resolved["stage_d_run"] = stage_d_run
    run_dir = begin_run(args.out_dir, resolved, run_id=args.run_id)

    manifest = pd.read_csv(stage_d_run / "sample_manifest.csv")
    inventory = pd.read_csv(args.inventory)
    predictions, metrics, summary = run_stage_e_garch(
        manifest,
        inventory,
        history=args.history,
        minimum_history=args.minimum_history,
    )
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    metrics.to_csv(run_dir / "validation_metrics.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    pooled = metrics[(metrics["group_type"] == "pooled") & (metrics["horizon"] == "path")]
    positive = metrics[(metrics["group_type"] == "label") & (metrics["group_value"] == "positive") & (metrics["horizon"] == "path")]
    control = metrics[(metrics["group_type"] == "label") & (metrics["group_value"] == "control") & (metrics["horizon"] == "path")]
    print(json.dumps(summary, indent=2))
    if not pooled.empty:
        print("\nPath metrics:")
        print(pd.concat([pooled, positive, control]).to_string(index=False))


if __name__ == "__main__":
    main()
