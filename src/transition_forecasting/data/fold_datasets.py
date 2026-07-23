from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.chronological_rematched_dataset import (
    build_rematched_rolling_dataset,
)
from transition_forecasting.modeling.stage_d_candidate_pool import (
    build_candidate_pool_from_series,
)

DEFAULT_N_FOLDS = 8
DEFAULT_TEST_FRACTION = 0.17
DEFAULT_EMBARGO_DAYS = 10
DEFAULT_CONTROLS_PER_POSITIVE = 3


def _unicode_array(values: pd.Series) -> np.ndarray:
    """Return a pickle-free fixed-width Unicode array."""
    strings = values.astype(str).tolist()
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"U{width}")


def _load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(dataset_dir / "sample_manifest.csv")
    with np.load(dataset_dir / "sequence_tensors.npz", allow_pickle=False) as archive:
        tensor = np.asarray(archive["X"])
        tensor_ids = archive["sample_id"].astype(str)
    manifest_ids = manifest["sample_id"].astype(str).to_numpy()
    if len(manifest) != len(tensor):
        raise ValueError("sample manifest and sequence tensor lengths differ")
    if not np.array_equal(manifest_ids, tensor_ids):
        raise ValueError("sample manifest and sequence tensor sample IDs are not aligned")
    if tensor.ndim != 3 or tensor.shape[2] != 1:
        raise ValueError(
            "canonical transition fold builder requires one-channel level tensors"
        )
    return manifest, tensor


def _load_daily_series(dataset_dir: Path) -> dict[str, pd.Series]:
    frame = pd.read_csv(dataset_dir / "daily_volatility.csv.gz")
    required = {"index", "date", "log_parkinson_volatility"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily volatility file is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.duplicated(["index", "date"]).any():
        raise ValueError("daily volatility contains duplicate index/date rows")

    result: dict[str, pd.Series] = {}
    for index_name, group in frame.groupby("index", sort=True):
        ordered = group.sort_values("date")
        values = ordered["log_parkinson_volatility"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"daily volatility contains non-finite values for {index_name}"
            )
        result[str(index_name)] = pd.Series(
            values,
            index=ordered["date"],
            name=str(index_name),
        )
    return result


def build_candidate_pool_from_dataset(
    dataset_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Rebuild the full eligible binary control pool from the frozen 1D dataset."""
    dataset_dir = Path(dataset_dir).resolve()
    catalogue = pd.read_csv(dataset_dir / "transition_catalogue.csv")
    required = {"index", "onset_date"}
    missing = required.difference(catalogue.columns)
    if missing:
        raise ValueError(f"transition catalogue is missing columns: {sorted(missing)}")
    catalogue["onset_date"] = pd.to_datetime(
        catalogue["onset_date"], errors="raise"
    )
    series_by_index = _load_daily_series(dataset_dir)

    frames: list[pd.DataFrame] = []
    tensors: list[np.ndarray] = []
    rows_by_index: dict[str, int] = {}
    for index_name, series in series_by_index.items():
        events = catalogue[catalogue["index"].astype(str).eq(index_name)]
        onset_positions = series.index.get_indexer(events["onset_date"])
        frame, tensor = build_candidate_pool_from_series(
            index_name=index_name,
            series=series,
            onset_positions=onset_positions,
        )
        frames.append(frame)
        tensors.append(tensor)
        rows_by_index[index_name] = int(len(frame))

    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tensor = (
        np.concatenate(tensors, axis=0)
        if tensors
        else np.empty((0, 40, 1), dtype=float)
    )
    if len(manifest) != len(tensor):
        raise ValueError("candidate manifest and tensor lengths differ")
    if not manifest.empty and manifest["candidate_id"].duplicated().any():
        raise ValueError("candidate IDs are not unique")
    manifest["_candidate_row"] = np.arange(len(manifest), dtype=int)

    summary = {
        "candidate_rows": int(len(manifest)),
        "unique_control_origins": int(
            manifest[["index", "origin_date"]].drop_duplicates().shape[0]
        )
        if len(manifest)
        else 0,
        "indices": int(manifest["index"].nunique()) if len(manifest) else 0,
        "leads": sorted(int(value) for value in manifest["lead"].unique())
        if len(manifest)
        else [],
        "sequence_shape": list(tensor.shape),
        "rows_by_index": rows_by_index,
        "source_dataset": str(dataset_dir),
        "control_protocol": "precontrol_binary_matching",
    }
    return manifest, tensor, summary


def _write_candidate_pool(
    dataset_dir: Path,
    manifest: pd.DataFrame,
    tensor: np.ndarray,
    summary: dict[str, object],
) -> None:
    persisted = manifest.drop(columns=["_candidate_row"], errors="ignore")
    persisted.to_csv(dataset_dir / "control_candidate_manifest.csv", index=False)
    np.savez_compressed(
        dataset_dir / "control_candidate_tensors.npz",
        X=tensor,
        candidate_id=_unicode_array(persisted["candidate_id"]),
        index=_unicode_array(persisted["index"]),
        lead=persisted["lead"].astype(int).to_numpy(),
    )
    (dataset_dir / "candidate_pool_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_fold_dataset(
    output_dir: Path,
    manifest: pd.DataFrame,
    tensor: np.ndarray,
    audit: pd.DataFrame,
    summary: dict[str, object],
    *,
    force: bool,
) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing folds: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest.to_csv(output_dir / "rematched_rolling_manifest.csv", index=False)
    audit.to_csv(output_dir / "control_match_audit.csv", index=False)
    np.savez_compressed(
        output_dir / "rematched_rolling_tensors.npz",
        X=tensor,
        sample_id=_unicode_array(manifest["sample_id"]),
        fold=manifest["fold"].astype(int).to_numpy(),
        fold_split=_unicode_array(manifest["fold_split"]),
        channel_names=np.asarray(["log_volatility_level"], dtype="U32"),
    )
    payload = dict(summary)
    payload["channels"] = ["log_volatility_level"]
    payload["output_dir"] = str(output_dir)
    payload["three_channel_dataset_built"] = False
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def build_one_channel_folds(
    dataset_1d_dir: Path,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    controls_per_positive: int = DEFAULT_CONTROLS_PER_POSITIVE,
    force: bool = False,
) -> dict[str, object]:
    """Build the canonical one-channel pre-control candidate pool and folds."""
    dataset_1d_dir = Path(dataset_1d_dir).resolve()
    manifest_1d, tensor_1d = _load_dataset(dataset_1d_dir)

    candidate_manifest, candidate_1d, candidate_summary = (
        build_candidate_pool_from_dataset(dataset_1d_dir)
    )
    _write_candidate_pool(
        dataset_1d_dir,
        candidate_manifest,
        candidate_1d,
        candidate_summary,
    )

    manifest_fold, tensor_fold, audit, fold_summary = (
        build_rematched_rolling_dataset(
            manifest_1d,
            tensor_1d,
            candidate_manifest,
            candidate_1d,
            n_folds=n_folds,
            test_fraction=test_fraction,
            embargo_days=embargo_days,
            controls_per_positive=controls_per_positive,
        )
    )

    output_1d = dataset_1d_dir / "purged_walk_forward_folds"
    _write_fold_dataset(
        output_1d,
        manifest_fold,
        tensor_fold,
        audit,
        fold_summary,
        force=force,
    )

    return {
        "dataset_1d": str(dataset_1d_dir),
        "candidate_rows": int(len(candidate_manifest)),
        "folds": int(n_folds),
        "test_fraction": float(test_fraction),
        "embargo_days": int(embargo_days),
        "controls_per_positive": int(controls_per_positive),
        "fold_output_1d": str(output_1d),
        "fold_manifest_rows": int(len(manifest_fold)),
        "fold_tensor_1d_shape": list(tensor_fold.shape),
        "channels": ["log_volatility_level"],
        "three_channel_dataset_built": False,
        "test_evaluated": False,
    }


def build_one_and_three_channel_folds(
    dataset_1d_dir: Path,
    dataset_3d_dir: Path | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """Compatibility wrapper; the redundant three-channel construction is disabled."""
    report = build_one_channel_folds(dataset_1d_dir, **kwargs)
    report["ignored_dataset_3d"] = (
        str(Path(dataset_3d_dir).resolve()) if dataset_3d_dir is not None else None
    )
    return report
