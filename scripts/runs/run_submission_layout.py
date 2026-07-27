from pathlib import Path


def transition_paths(run_dir: Path) -> dict[str, Path]:
    root = run_dir / "files"
    data = root / "data"
    raw = data / "raw" / "global_stock_indices_historical_data"
    processed = data / "processed"
    dataset = processed / "global_transition_dataset_1d"
    data_validation = data / "validation"
    classical = root / "classical_baselines"
    classical_validation = classical / "validation"
    return {
        "root": root,
        "data_root": data,
        "raw": raw,
        "dataset_1d": dataset,
        "folds_1d": dataset / "purged_walk_forward_folds",
        "validation": data_validation / "data_pipeline_audit.json",
        "checksums": data_validation / "data_pipeline_checksums.json",
        "classical_root": classical,
        "linear_results_root": classical / "linear" / "run",
        "garch_results_root": classical / "garch" / "run",
        "esn_results_root": classical / "esn" / "run",
        "canonical_results_root": classical / "canonical" / "run",
        "classical_validation": (
            classical_validation / "classical_baseline_audit.json"
        ),
    }
