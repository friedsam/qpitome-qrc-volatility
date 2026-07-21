from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 10
WINDOW = 40
TARGET_COLUMNS = [f"target_x_h{i}" for i in range(1, HORIZON + 1)]
VAL_START = pd.Timestamp("2013-01-01", tz="UTC")
TEST_START = pd.Timestamp("2016-01-01", tz="UTC")


def qlike(y_true_logvol: np.ndarray, y_pred_logvol: np.ndarray) -> np.ndarray:
    true_logvar = np.clip(2.0 * np.asarray(y_true_logvol, dtype=float), -40.0, 20.0)
    pred_logvar = np.clip(2.0 * np.asarray(y_pred_logvol, dtype=float), -40.0, 20.0)
    ratio = np.exp(np.clip(true_logvar - pred_logvar, -40.0, 40.0))
    return ratio - np.log(ratio) - 1.0


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x, train_y)
    return np.asarray(model.predict(eval_x), dtype=float)


def har_features(sequence: np.ndarray) -> np.ndarray:
    level = sequence[:, -1]
    return np.column_stack(
        [level, sequence[:, -5:].mean(axis=1), sequence[:, -20:].mean(axis=1)]
    )


def metric_records(
    *,
    dataset: str,
    model: str,
    fold: int | str,
    frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, object]]:
    groups: list[tuple[str, str, np.ndarray]] = [
        ("pooled", "all", np.ones(len(frame), dtype=bool))
    ]
    if "label" in frame.columns:
        labels = frame["label"].to_numpy()
        groups.extend(
            [
                ("label", "transition", labels == 1),
                ("label", "control", labels == 0),
            ]
        )
    if "lead" in frame.columns:
        leads = pd.to_numeric(frame["lead"], errors="coerce").to_numpy()
        for lead in sorted(pd.Series(leads).dropna().unique()):
            lead_mask = leads == lead
            groups.append(("lead", str(int(lead)), lead_mask))
            if "label" in frame.columns:
                labels = frame["label"].to_numpy()
                groups.extend(
                    [
                        (
                            "lead_label",
                            f"L{int(lead)}_transition",
                            lead_mask & (labels == 1),
                        ),
                        (
                            "lead_label",
                            f"L{int(lead)}_control",
                            lead_mask & (labels == 0),
                        ),
                    ]
                )

    rows: list[dict[str, object]] = []
    for group_type, group_value, mask in groups:
        if not mask.any():
            continue
        yt = actual[mask]
        yp = predicted[mask]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "fold": fold,
                "group_type": group_type,
                "group_value": group_value,
                "horizon": "path",
                "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "qlike": float(qlike(yt, yp).mean()),
            }
        )
        for horizon in range(HORIZON):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "fold": fold,
                    "group_type": group_type,
                    "group_value": group_value,
                    "horizon": horizon + 1,
                    "n": int(mask.sum()),
                    "rmse": float(
                        np.sqrt(np.mean((yt[:, horizon] - yp[:, horizon]) ** 2))
                    ),
                    "qlike": float(
                        qlike(yt[:, horizon], yp[:, horizon]).mean()
                    ),
                }
            )
    return rows


def prediction_frame(
    *,
    dataset: str,
    model: str,
    fold: int | str,
    frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    identity = [
        column
        for column in [
            "sample_id",
            "episode_id",
            "matched_positive_id",
            "label",
            "lead",
            "index",
            "ticker",
            "origin_date",
        ]
        if column in frame.columns
    ]
    output = frame[identity].copy()
    output.insert(0, "dataset", dataset)
    output.insert(1, "model", model)
    output.insert(2, "fold", fold)
    for horizon in range(HORIZON):
        output[f"actual_h{horizon + 1}"] = actual[:, horizon]
        output[f"predicted_h{horizon + 1}"] = predicted[:, horizon]
    return output


def load_frozen_folds(fold_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest_path = fold_dir / "rematched_rolling_manifest.csv"
    tensor_path = fold_dir / "rematched_rolling_tensors.npz"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError(
            "Frozen fold artifacts are missing. Expected both "
            f"{manifest_path} and {tensor_path}."
        )
    manifest = pd.read_csv(manifest_path)
    missing_targets = sorted(set(TARGET_COLUMNS) - set(manifest.columns))
    if missing_targets:
        raise ValueError(f"fold manifest is missing targets: {missing_targets}")
    with np.load(tensor_path, allow_pickle=False) as archive:
        tensor = np.asarray(archive["X"], dtype=float)
        tensor_ids = archive["sample_id"].astype(str)
    if tensor.ndim == 3 and tensor.shape[2] == 1:
        tensor = tensor[:, :, 0]
    if tensor.ndim != 2 or tensor.shape[1] != WINDOW:
        raise ValueError(f"expected a ({WINDOW},) one-channel sequence per row, got {tensor.shape}")
    manifest_ids = manifest["sample_id"].astype(str).to_numpy()
    if len(manifest) != len(tensor) or not np.array_equal(manifest_ids, tensor_ids):
        raise ValueError("fold manifest and tensor archive are not aligned")
    return manifest, tensor


def run_frozen_fold_baselines(
    fold_dir: Path,
    *,
    alpha: float,
    shuffled_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest, sequence = load_frozen_folds(fold_dir)
    targets = manifest[TARGET_COLUMNS].to_numpy(dtype=float)
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    fold_counts: list[dict[str, object]] = []

    for fold in sorted(manifest["fold"].astype(int).unique()):
        fold_mask = manifest["fold"].astype(int).to_numpy() == fold
        train = fold_mask & manifest["fold_split"].eq("train").to_numpy()
        valid = fold_mask & manifest["fold_split"].eq("val").to_numpy()
        test = fold_mask & manifest["fold_split"].eq("test").to_numpy()
        if not train.any() or not valid.any() or not test.any():
            raise ValueError(f"fold {fold} lacks train, validation, or reserved test rows")

        rng = np.random.default_rng(shuffled_seed + fold)
        shuffled_train = sequence[train].copy()
        shuffled_valid = sequence[valid].copy()
        for row in shuffled_train:
            rng.shuffle(row)
        for row in shuffled_valid:
            rng.shuffle(row)

        predictions = {
            "persistence": np.repeat(sequence[valid, -1][:, None], HORIZON, axis=1),
            "har": fit_ridge(
                har_features(sequence[train]),
                targets[train],
                har_features(sequence[valid]),
                alpha=alpha,
            ),
            "raw_sequence_ridge": fit_ridge(
                sequence[train], targets[train], sequence[valid], alpha=alpha
            ),
            "raw_sequence_ridge_shuffled": fit_ridge(
                shuffled_train, targets[train], shuffled_valid, alpha=alpha
            ),
        }
        valid_frame = manifest.loc[valid].reset_index(drop=True)
        valid_targets = targets[valid]
        for model, prediction in predictions.items():
            all_metrics.extend(
                metric_records(
                    dataset="frozen_transition_folds",
                    model=model,
                    fold=int(fold),
                    frame=valid_frame,
                    actual=valid_targets,
                    predicted=prediction,
                )
            )
            all_predictions.append(
                prediction_frame(
                    dataset="frozen_transition_folds",
                    model=model,
                    fold=int(fold),
                    frame=valid_frame,
                    actual=valid_targets,
                    predicted=prediction,
                )
            )
        fold_counts.append(
            {
                "fold": int(fold),
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "reserved_test_rows": int(test.sum()),
                "test_predictions_written": 0,
            }
        )

    metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    path = metrics[metrics["horizon"].eq("path")]
    aggregate = (
        path.groupby(["model", "group_type", "group_value"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            total_validation_rows=("n", "sum"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_qlike=("qlike", "mean"),
            sd_qlike=("qlike", "std"),
        )
        .to_dict("records")
    )
    return metrics, predictions, {
        "fold_counts": fold_counts,
        "aggregate_path_metrics": aggregate,
    }


def log_parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    volatility = np.abs(np.log(high / low)) / np.sqrt(4.0 * np.log(2.0))
    return np.log(volatility.replace(0.0, np.nan))


def build_broad_samples(
    panel_path: Path,
    *,
    stride: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not panel_path.is_file():
        raise FileNotFoundError(f"Field-experiment panel is missing: {panel_path}")
    raw = pd.read_csv(panel_path, usecols=["date", "high", "low", "ticker"])
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.normalize()
    raw["high"] = pd.to_numeric(raw["high"], errors="coerce")
    raw["low"] = pd.to_numeric(raw["low"], errors="coerce")
    raw = raw[
        (raw["high"] > 0)
        & (raw["low"] > 0)
        & (raw["high"] >= raw["low"])
    ].copy()
    raw["x"] = log_parkinson(raw["high"], raw["low"])
    raw = (
        raw.dropna(subset=["x"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
    )

    rows: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for ticker, group in raw.groupby("ticker", sort=True):
        series = group.set_index("date")["x"].sort_index()
        values = series.to_numpy(dtype=float)
        dates = series.index
        for origin in range(WINDOW - 1, len(series) - HORIZON, stride):
            sequence = values[origin - WINDOW + 1 : origin + 1]
            target = values[origin + 1 : origin + 1 + HORIZON]
            if not np.isfinite(sequence).all() or not np.isfinite(target).all():
                continue
            date = dates[origin]
            split = (
                "train"
                if date < VAL_START
                else "val"
                if date < TEST_START
                else "test"
            )
            rows.append(
                {
                    "sample_id": f"B_{ticker}_{date.date().isoformat()}",
                    "ticker": str(ticker),
                    "origin_date": date,
                    "split": split,
                }
            )
            sequences.append(sequence)
            targets.append(target)
    if not rows:
        raise ValueError("no broad chronological samples were built")
    return pd.DataFrame(rows), np.asarray(sequences), np.asarray(targets)


def run_field_reproduction(
    panel_path: Path,
    *,
    alpha: float,
    stride: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    frame, sequence, targets = build_broad_samples(panel_path, stride=stride)
    train = frame["split"].eq("train").to_numpy()
    valid = frame["split"].eq("val").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    models = {
        "persistence": np.repeat(sequence[valid, -1][:, None], HORIZON, axis=1),
        "har": fit_ridge(
            har_features(sequence[train]),
            targets[train],
            har_features(sequence[valid]),
            alpha=alpha,
        ),
        "raw_sequence_ridge": fit_ridge(
            sequence[train], targets[train], sequence[valid], alpha=alpha
        ),
    }
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    valid_frame = frame.loc[valid].reset_index(drop=True)
    for model, prediction in models.items():
        metrics.extend(
            metric_records(
                dataset="broad_chronological_field_reproduction",
                model=model,
                fold="field",
                frame=valid_frame,
                actual=targets[valid],
                predicted=prediction,
            )
        )
        predictions.append(
            prediction_frame(
                dataset="broad_chronological_field_reproduction",
                model=model,
                fold="field",
                frame=valid_frame,
                actual=targets[valid],
                predicted=prediction,
            )
        )
    counts = {
        "train": int(train.sum()),
        "validation": int(valid.sum()),
        "reserved_test": int(test.sum()),
        "test_predictions_written": 0,
    }
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), counts


def compare_with_reference(
    metrics: pd.DataFrame,
    reference_path: Path,
) -> dict[str, object]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference["broad_chronological"]
    tolerances = reference["reproduction_tolerances"]
    path = metrics[
        metrics["dataset"].eq("broad_chronological_field_reproduction")
        & metrics["horizon"].eq("path")
        & metrics["group_type"].eq("pooled")
    ].set_index("model")

    checks: list[dict[str, object]] = []
    for model, expected_metrics in expected["path_metrics"].items():
        row = path.loc[model]
        for metric, tolerance_key in [
            ("rmse", "rmse_absolute"),
            ("qlike", "qlike_absolute"),
        ]:
            observed = float(row[metric])
            target = float(expected_metrics[metric])
            tolerance = float(tolerances[tolerance_key])
            checks.append(
                {
                    "check": f"field_{model}_{metric}",
                    "observed": observed,
                    "expected": target,
                    "absolute_difference": abs(observed - target),
                    "tolerance": tolerance,
                    "passed": abs(observed - target) <= tolerance,
                }
            )
    observed_n = int(path.loc["har", "n"])
    expected_n = int(expected["validation_samples"])
    checks.append(
        {
            "check": "field_validation_sample_count",
            "observed": observed_n,
            "expected": expected_n,
            "passed": observed_n == expected_n,
        }
    )
    checks.extend(
        [
            {
                "check": "field_har_beats_persistence_rmse",
                "passed": float(path.loc["har", "rmse"])
                < float(path.loc["persistence", "rmse"]),
            },
            {
                "check": "field_raw_ridge_beats_har_rmse",
                "passed": float(path.loc["raw_sequence_ridge", "rmse"])
                < float(path.loc["har", "rmse"]),
            },
        ]
    )
    return {
        "reference_path": str(reference_path),
        "checks": checks,
        "all_checks_passed": all(bool(check["passed"]) for check in checks),
    }


def frozen_sanity(metrics: pd.DataFrame) -> dict[str, object]:
    path = metrics[
        metrics["dataset"].eq("frozen_transition_folds")
        & metrics["horizon"].eq("path")
    ]
    aggregate = (
        path.groupby(["model", "group_type", "group_value"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_qlike=("qlike", "mean"))
    )

    def value(model: str, group_type: str, group_value: str, metric: str) -> float:
        match = aggregate[
            aggregate["model"].eq(model)
            & aggregate["group_type"].eq(group_type)
            & aggregate["group_value"].eq(group_value)
        ]
        if len(match) != 1:
            raise ValueError(
                f"missing unique aggregate metric for {model}/{group_type}/{group_value}"
            )
        return float(match.iloc[0][metric])

    checks = [
        {
            "check": "har_transition_qlike_exceeds_control",
            "passed": value("har", "label", "transition", "mean_qlike")
            > value("har", "label", "control", "mean_qlike"),
        },
        {
            "check": "har_transition_rmse_exceeds_control",
            "passed": value("har", "label", "transition", "mean_rmse")
            > value("har", "label", "control", "mean_rmse"),
        },
        {
            "check": "ordered_raw_ridge_beats_shuffled_transition_qlike",
            "passed": value(
                "raw_sequence_ridge", "label", "transition", "mean_qlike"
            )
            < value(
                "raw_sequence_ridge_shuffled",
                "label",
                "transition",
                "mean_qlike",
            ),
        },
    ]
    return {
        "checks": checks,
        "all_checks_passed": all(bool(check["passed"]) for check in checks),
        "interpretation": (
            "Failure is a scientific result unless caused by an implementation "
            "or data-integrity error."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run primitive classical baselines on frozen folds and reproduce "
            "the field experiment."
        )
    )
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "config/transition_forecasting/classical_baselines/"
            "field_experiment_reference.json"
        ),
    )
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--broad-stride", type=int, default=1)
    parser.add_argument("--shuffled-seed", type=int, default=20260720)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=False)

    fold_metrics, fold_predictions, fold_summary = run_frozen_fold_baselines(
        args.fold_dir,
        alpha=args.alpha,
        shuffled_seed=args.shuffled_seed,
    )
    field_metrics, field_predictions, field_counts = run_field_reproduction(
        args.panel,
        alpha=args.alpha,
        stride=args.broad_stride,
    )
    metrics = pd.concat([fold_metrics, field_metrics], ignore_index=True)
    predictions = pd.concat(
        [fold_predictions, field_predictions], ignore_index=True, sort=False
    )

    metrics.to_csv(args.out_dir / "metrics_by_fold.csv", index=False)
    metrics[metrics["horizon"].ne("path")].to_csv(
        args.out_dir / "metrics_by_horizon.csv", index=False
    )
    metrics[
        metrics["horizon"].eq("path") & metrics["group_type"].ne("pooled")
    ].to_csv(args.out_dir / "metrics_by_group.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)

    field_comparison = compare_with_reference(metrics, args.reference)
    frozen_checks = frozen_sanity(metrics)
    comparison_payload = {
        "test_evaluated": False,
        "field_reproduction": field_comparison,
        "frozen_fold_sanity": frozen_checks,
    }
    (args.out_dir / "field_experiment_comparison.json").write_text(
        json.dumps(comparison_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "test_evaluated": False,
        "alpha": args.alpha,
        "broad_stride": args.broad_stride,
        "shuffled_seed": args.shuffled_seed,
        "fold_dir": str(args.fold_dir),
        "panel": str(args.panel),
        "fold_summary": fold_summary,
        "field_counts": field_counts,
        "field_reproduction_passed": field_comparison["all_checks_passed"],
        "frozen_directional_sanity_passed": frozen_checks["all_checks_passed"],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["field_reproduction_passed"] else 1
