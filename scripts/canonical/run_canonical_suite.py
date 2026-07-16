#!/usr/bin/env python3
"""Run or assemble a selectable canonical model comparison.

Models without ``--source`` overrides are executed through their own scientific
runner. The default remains ``classical-all``; quantum models are opt-in.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MASTER_SCRIPT = SCRIPT_DIR / "run_master_comparison.py"
TFIM_SCRIPT = SCRIPT_DIR / "run_canonical_tfim.py"
RYDBERG_SCRIPT = SCRIPT_DIR / "run_canonical_rydberg.py"
LSTM_SCRIPT = REPO_ROOT / "scripts/baselines/lstm/run_phase3_lstm_walkforward.py"
GARCH_SCRIPT = REPO_ROOT / "scripts/baselines/garch/run_phase3_garch_walkforward.py"

MASTER_MODELS = (
    "persistence_20d",
    "har_ridge",
    "raw_ridge",
    "esn_selected",
)
CLASSICAL_STANDALONE_MODELS = ("garch_1_1_t", "lstm")
TFIM_MODELS = ("tfim_phase2_final",)
RYDBERG_MODELS = (
    "rydberg_temporal",
    "rydberg_memoryless",
    "rydberg_shuffled",
    "rydberg_multi_lb",
    "rydberg_multi_lb_memoryless",
    "rydberg_multi_lb_shuffled",
)
QUANTUM_MODELS = TFIM_MODELS + RYDBERG_MODELS
STANDALONE_MODELS = CLASSICAL_STANDALONE_MODELS + QUANTUM_MODELS
AVAILABLE_MODELS = MASTER_MODELS + STANDALONE_MODELS
CLASSICAL_ALL = MASTER_MODELS + CLASSICAL_STANDALONE_MODELS
MODEL_GROUPS = {
    "classical-core": MASTER_MODELS,
    "classical-all": CLASSICAL_ALL,
    "tfim": TFIM_MODELS,
    "rydberg": RYDBERG_MODELS,
    "quantum": QUANTUM_MODELS,
    "classical+quantum": CLASSICAL_ALL + QUANTUM_MODELS,
    "all-available": AVAILABLE_MODELS,
}


def load_master_module():
    spec = importlib.util.spec_from_file_location("canonical_master", MASTER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MASTER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", default="canonical")
    parser.add_argument("--groups", nargs="*", choices=sorted(MODEL_GROUPS), default=None)
    parser.add_argument("--models", nargs="*", choices=sorted(AVAILABLE_MODELS), default=None)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="MODEL=RUN_DIRECTORY",
    )
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--n-folds", type=int, default=7)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--only-folds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/canonical/run_canonical_suite",
    )
    return parser.parse_args()


def selected_models(args: argparse.Namespace) -> list[str]:
    groups = args.groups
    models = args.models
    if groups is None and models is None:
        groups = ["classical-all"]
    selected: list[str] = []
    for group in groups or []:
        for model in MODEL_GROUPS[group]:
            if model not in selected:
                selected.append(model)
    for model in models or []:
        if model not in selected:
            selected.append(model)
    if not selected:
        raise ValueError("No models selected")
    return selected


def parse_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --source {value!r}; expected MODEL=RUN_DIRECTORY")
        model, raw_path = value.split("=", 1)
        model = model.strip()
        if model not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown source model {model!r}")
        if model in sources:
            raise ValueError(f"Duplicate --source for {model}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_dir():
            raise FileNotFoundError(f"Source run directory not found for {model}: {path}")
        sources[model] = path.resolve()
    return sources


def unique_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern!r} in {directory}; found {len(matches)}"
        )
    return matches[0]


def read_run_parameters(directory: Path) -> tuple[dict, str | None]:
    path = directory / "params.json"
    if not path.exists():
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload.get("parameters", {})
    return parameters if isinstance(parameters, dict) else {}, str(path)


def compatibility_report(
    args: argparse.Namespace,
    model: str,
    directory: Path,
    metrics: pd.DataFrame,
) -> dict:
    source_params, params_path = read_run_parameters(directory)
    requested = {
        "n_folds": args.n_folds,
        "min_train": args.min_train,
        "val_size": args.val_size,
        "purge": args.purge,
        "only_folds": args.only_folds,
    }
    if model != "garch_1_1_t":
        requested["lookback"] = args.lookback
    source = {key: source_params.get(key) for key in requested}
    differences = {
        key: {"requested": requested[key], "source": source[key]}
        for key in requested
        if source[key] is not None and source[key] != requested[key]
    }
    missing = [key for key, value in source.items() if value is None]
    actual_folds = sorted(
        int(value)
        for value in pd.to_numeric(metrics["fold"], errors="coerce").dropna().unique()
    )
    expected_folds = (
        sorted(int(value) for value in args.only_folds)
        if args.only_folds
        else list(range(1, args.n_folds + 1))
    )
    if actual_folds != expected_folds:
        differences["actual_fold_ids"] = {
            "requested": expected_folds,
            "source": actual_folds,
        }
    status = (
        "different"
        if differences
        else "incomplete"
        if missing or params_path is None
        else "passed"
    )
    return {
        "status": status,
        "params_file": params_path,
        "requested": requested,
        "source": source,
        "actual_fold_ids": actual_folds,
        "actual_fold_count": len(actual_folds),
        "missing_parameter_fields": missing,
        "differences": differences,
        "policy": "record differences without blocking deliberate mixed-source comparisons",
    }


def run_child(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Child run failed with exit code {completed.returncode}; see {log_path}"
        )


def common_fold_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--n-folds", str(args.n_folds),
        "--min-train", str(args.min_train),
        "--val-size", str(args.val_size),
        "--purge", str(args.purge),
    ]
    if args.only_folds:
        values.extend(["--only-folds", *[str(value) for value in args.only_folds]])
    return values


def generated_sources(
    args: argparse.Namespace,
    models: list[str],
    overrides: dict[str, Path],
) -> tuple[dict[str, Path], list[dict]]:
    sources = dict(overrides)
    commands: list[dict] = []
    missing = [model for model in models if model not in sources]
    if not missing:
        return sources, commands
    if args.no_run:
        raise ValueError(
            "--no-run requires --source for every selected model; missing: "
            + ", ".join(missing)
        )

    child_run_id = args.out_dir.name

    master_models = [model for model in missing if model in MASTER_MODELS]
    if master_models:
        command = [
            sys.executable,
            str(MASTER_SCRIPT.relative_to(REPO_ROOT)),
            "--run-id", child_run_id,
            "--tag", args.tag,
            "--models", *master_models,
            "--lookback", str(args.lookback),
            *common_fold_args(args),
        ]
        log_path = args.out_dir / "child_master.log"
        run_child(command, log_path)
        run_dir = REPO_ROOT / "results/canonical/run_master_comparison" / child_run_id
        for model in master_models:
            sources[model] = run_dir
        commands.append({"models": master_models, "command": command, "log": str(log_path)})

    if "tfim_phase2_final" in missing:
        command = [
            sys.executable,
            str(TFIM_SCRIPT.relative_to(REPO_ROOT)),
            "--run-id", child_run_id,
            "--tag", "tfim_phase2_final",
            "--lookback", str(args.lookback),
            *common_fold_args(args),
        ]
        log_path = args.out_dir / "child_tfim.log"
        run_child(command, log_path)
        sources["tfim_phase2_final"] = (
            REPO_ROOT / "results/canonical/run_canonical_tfim" / child_run_id
        )
        commands.append({"models": ["tfim_phase2_final"], "command": command, "log": str(log_path)})

    rydberg_models = [model for model in missing if model in RYDBERG_MODELS]
    if rydberg_models:
        command = [
            sys.executable,
            str(RYDBERG_SCRIPT.relative_to(REPO_ROOT)),
            "--run-id", child_run_id,
            "--tag", "rydberg_historical",
            "--models", *rydberg_models,
            "--lookback", str(args.lookback),
            *common_fold_args(args),
        ]
        log_path = args.out_dir / "child_rydberg.log"
        run_child(command, log_path)
        run_dir = REPO_ROOT / "results/canonical/run_canonical_rydberg" / child_run_id
        for model in rydberg_models:
            sources[model] = run_dir
        commands.append({"models": rydberg_models, "command": command, "log": str(log_path)})

    if "lstm" in missing:
        command = [
            sys.executable,
            str(LSTM_SCRIPT.relative_to(REPO_ROOT)),
            "--run-id", child_run_id,
            "--tag", args.tag,
            "--lookback", str(args.lookback),
            *common_fold_args(args),
        ]
        log_path = args.out_dir / "child_lstm.log"
        run_child(command, log_path)
        sources["lstm"] = (
            REPO_ROOT / "results/baselines/lstm/run_phase3_lstm_walkforward" / child_run_id
        )
        commands.append({"models": ["lstm"], "command": command, "log": str(log_path)})

    if "garch_1_1_t" in missing:
        command = [
            sys.executable,
            str(GARCH_SCRIPT.relative_to(REPO_ROOT)),
            "--run-id", child_run_id,
            "--tag", args.tag,
            *common_fold_args(args),
        ]
        log_path = args.out_dir / "child_garch.log"
        run_child(command, log_path)
        sources["garch_1_1_t"] = (
            REPO_ROOT / "results/baselines/garch/run_phase3_garch_walkforward" / child_run_id
        )
        commands.append({"models": ["garch_1_1_t"], "command": command, "log": str(log_path)})

    return sources, commands


def empty_classification_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for quantile in (90, 95):
        for metric in ("ap", "auc", "f1", "precision", "recall", "called_rate"):
            output[f"q{quantile}_test_{metric}"] = np.nan
    return output


def normalize_lstm_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame.loc[frame["split"] == "test"].copy()
    if test.empty:
        raise ValueError("LSTM metrics contain no test rows")
    test["protocol"] = "classical"
    test = test.rename(columns={
        "rmse": "test_rmse",
        "qlike": "test_qlike",
        "mz_alpha": "test_mz_alpha",
        "mz_beta": "test_mz_beta",
        "mz_r2": "test_mz_r2",
    })
    return empty_classification_columns(test)


def normalize_garch_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame.loc[(frame["split"] == "test") & (frame["task"] == "level")].copy()
    if test.empty:
        raise ValueError("GARCH metrics contain no level-task test rows")
    test["protocol"] = "classical"
    test = test.rename(columns={
        "rmse": "test_rmse",
        "qlike": "test_qlike",
        "mz_alpha": "test_mz_alpha",
        "mz_beta": "test_mz_beta",
        "mz_r2": "test_mz_r2",
    })
    return empty_classification_columns(test)


def normalize_lstm_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["protocol"] = "classical"
    output["log_score"] = np.log(np.clip(output["y_pred"].to_numpy(float), 1e-12, None))
    output["q90_label"] = np.nan
    output["q95_label"] = np.nan
    return output[[
        "fold", "model", "protocol", "split", "date", "y_true",
        "log_score", "y_pred", "q90_label", "q95_label",
    ]]


def normalize_garch_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[frame["task"] == "level"].copy()
    output["protocol"] = "classical"
    output["y_true"] = output["future_rv_true"]
    output["y_pred"] = output["predicted_future_rv"]
    output["log_score"] = np.log(np.clip(output["y_pred"].to_numpy(float), 1e-12, None))
    output["q90_label"] = np.nan
    output["q95_label"] = np.nan
    return output[[
        "fold", "model", "protocol", "split", "date", "y_true",
        "log_score", "y_pred", "q90_label", "q95_label",
    ]]


def load_model_result(
    model: str,
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if model in MASTER_MODELS or model in QUANTUM_MODELS:
        metrics_path = unique_match(directory, "per_fold_metrics_*.csv")
        predictions_path = unique_match(directory, "predictions_*.csv")
        metrics = pd.read_csv(metrics_path)
        predictions = pd.read_csv(predictions_path)
        metrics = metrics.loc[metrics["model"] == model].copy()
        predictions = predictions.loc[predictions["model"] == model].copy()
    elif model == "lstm":
        metrics_path = unique_match(directory, "lstm_metrics_*.csv")
        predictions_path = unique_match(directory, "lstm_predictions_*.csv")
        metrics = normalize_lstm_metrics(pd.read_csv(metrics_path))
        predictions = normalize_lstm_predictions(pd.read_csv(predictions_path))
    elif model == "garch_1_1_t":
        metrics_path = unique_match(directory, "garch_metrics_*.csv")
        predictions_path = unique_match(directory, "garch_predictions_*.csv")
        metrics = normalize_garch_metrics(pd.read_csv(metrics_path))
        predictions = normalize_garch_predictions(pd.read_csv(predictions_path))
    else:
        raise ValueError(model)
    if metrics.empty or predictions.empty:
        raise ValueError(f"No rows for {model} in {directory}")
    return metrics, predictions, {
        "model": model,
        "run_directory": str(directory),
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
    }


def main() -> None:
    args = parse_args()
    models = selected_models(args)
    overrides = parse_sources(args.source)
    unknown_overrides = sorted(set(overrides) - set(models))
    if unknown_overrides:
        raise ValueError(
            "Sources were supplied for unselected models: " + ", ".join(unknown_overrides)
        )

    args.out_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    sources, child_commands = generated_sources(args, models, overrides)

    metric_frames = []
    prediction_frames = []
    source_manifest = []
    for model in models:
        metrics, predictions, source = load_model_result(model, sources[model])
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        source["mode"] = "override" if model in overrides else "generated"
        source["compatibility"] = compatibility_report(
            args,
            model,
            sources[model],
            metrics,
        )
        source_manifest.append(source)

    combined_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    combined_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    master = load_master_module()
    aggregate = master.aggregate_metrics(combined_metrics)

    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    predictions_path = args.out_dir / f"predictions_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"

    combined_metrics.to_csv(per_fold_path, index=False)
    combined_predictions.to_csv(predictions_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "selected_groups": args.groups,
                "selected_models": models,
                "available_groups": {
                    key: list(value) for key, value in MODEL_GROUPS.items()
                },
                "sources": source_manifest,
                "child_commands": child_commands,
                "compatibility_policy": (
                    "Source differences are recorded per model and do not block "
                    "deliberate mixed-source comparisons."
                ),
                "classification_note": (
                    "Master, TFIM, and Rydberg runners include q90/q95 evaluation. "
                    "GARCH and LSTM currently contribute regression metrics only."
                ),
                "rydberg_status": (
                    "The six historical Rydberg models are opt-in reproducibility models. "
                    "They preserve the underdeveloped two-channel volatility encoding and "
                    "are not included in the default classical-all suite."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(aggregate.to_string(index=False))
    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
