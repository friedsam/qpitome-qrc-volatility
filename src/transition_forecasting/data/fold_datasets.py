from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.data.three_channel import CHANNEL_NAMES, to_three_channel
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
            raise ValueError(f"daily volatility contains non-finite values for {index_name}")
        result[str(index_name)] = pd.Series(values, index=ordered["date"], name=str(index_name))
    return result


def build_candidate_pool_from_dataset(
    dataset_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Rebuild the full eligible control pool from the frozen processed dataset."""
    dataset_dir = Path(dataset_dir).resolve()
    catalogue = pd.read_csv(dataset_dir / "transition_catalogue.csv")
    required = {"index", "onset_date"}
    missing = required.difference(catalogue.columns)
    if missing:
        raise ValueError(f"transition catalogue is missing columns: {sorted(missing)}")
    catalogue["onset_date"] = pd.to_datetime(catalogue["onset_date"], errors="raise")
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
    tensor = np.concatenate(tensors, axis=0) if tensors else np.empty((0, 40, 1), dtype=float)
    if len(manifest) != len(tensor):
        raise ValueError("candidate manifest and tensor lengths differ")
    if manifest["candidate_id"].duplicated().any():
        raise ValueError("candidate IDs are not unique")
    manifest["_candidate_row"] = np.arange(len(manifest), dtype=int)

    summary = {
        "candidate_rows": int(len(manifest)),
        "unique_control_origins": int(
            manifest[["index", "origin_date"]].drop_duplicates().shape[0]
        ),
        "indices": int(manifest["index"].nunique()),
        "leads": sorted(int(value) for value in manifest["lead"].unique()),
        "sequence_shape": list(tensor.shape),
        "rows_by_index": rows_by_index,
        "source_dataset": str(dataset_dir),
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
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def _write_fold_dataset(
    output_dir: Path,
    manifest: pd.DataFrame,
    tensor: np.ndarray,
    audit: pd.DataFrame,
    summary: dict[str, object],
    *,
    channel_names: list[str],
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
        channel_names=np.asarray(channel_names, dtype="U32"),
    )
    payload = dict(summary)
    payload["channels"] = list(channel_names)
    payload["output_dir"] = str(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def build_one_and_three_channel_folds(
    dataset_1d_dir: Path,
    dataset_3d_dir: Path,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    controls_per_positive: int = DEFAULT_CONTROLS_PER_POSITIVE,
    force: bool = False,
) -> dict[str, object]:
    """Build inspectable 1D and 3D candidate pools and identical fold assignments."""
    dataset_1d_dir = Path(dataset_1d_dir).resolve()
    dataset_3d_dir = Path(dataset_3d_dir).resolve()
    manifest_1d, tensor_1d = _load_dataset(dataset_1d_dir)
    manifest_3d, tensor_3d = _load_dataset(dataset_3d_dir)
    if not np.array_equal(
        manifest_1d["sample_id"].astype(str).to_numpy(),
        manifest_3d["sample_id"].astype(str).to_numpy(),
    ):
        raise ValueError("1D and 3D sample manifests are not aligned")
    expected_3d = to_three_channel(tensor_1d)
    if not np.array_equal(tensor_3d, expected_3d):
        raise ValueError("3D sequence tensor is not the deterministic transform of the 1D tensor")

    candidate_manifest, candidate_1d, candidate_summary = build_candidate_pool_from_dataset(
        dataset_1d_dir
    )
    candidate_3d = to_three_channel(candidate_1d)
    _write_candidate_pool(dataset_1d_dir, candidate_manifest, candidate_1d, candidate_summary)
    summary_3d = dict(candidate_summary)
    summary_3d["sequence_shape"] = list(candidate_3d.shape)
    summary_3d["channels"] = CHANNEL_NAMES.tolist()
    summary_3d["derived_from"] = str(dataset_1d_dir / "control_candidate_tensors.npz")
    _write_candidate_pool(dataset_3d_dir, candidate_manifest, candidate_3d, summary_3d)

    folds_1d = build_rematched_rolling_dataset(
        manifest_1d,
        tensor_1d,
        candidate_manifest,
        candidate_1d,
        n_folds=n_folds,
        test_fraction=test_fraction,
        embargo_days=embargo_days,
        controls_per_positive=controls_per_positive,
    )
    folds_3d = build_rematched_rolling_dataset(
        manifest_3d,
        tensor_3d,
        candidate_manifest,
        candidate_3d,
        n_folds=n_folds,
        test_fraction=test_fraction,
        embargo_days=embargo_days,
        controls_per_positive=controls_per_positive,
    )

    manifest_fold_1d, tensor_fold_1d, audit_1d, fold_summary_1d = folds_1d
    manifest_fold_3d, tensor_fold_3d, audit_3d, fold_summary_3d = folds_3d
    identity_columns = ["sample_id", "fold", "fold_split"]
    if not manifest_fold_1d[identity_columns].equals(manifest_fold_3d[identity_columns]):
        raise ValueError("1D and 3D fold manifests differ")
    if not audit_1d.equals(audit_3d):
        raise ValueError("1D and 3D control matching audits differ")
    if not np.array_equal(tensor_fold_3d, to_three_channel(tensor_fold_1d)):
        raise ValueError("3D fold tensor is not the deterministic transform of the 1D tensor")

    output_1d = dataset_1d_dir / "purged_walk_forward_folds"
    output_3d = dataset_3d_dir / "purged_walk_forward_folds"
    _write_fold_dataset(
        output_1d,
        manifest_fold_1d,
        tensor_fold_1d,
        audit_1d,
        fold_summary_1d,
        channel_names=["log_volatility_level"],
        force=force,
    )
    _write_fold_dataset(
        output_3d,
        manifest_fold_3d,
        tensor_fold_3d,
        audit_3d,
        fold_summary_3d,
        channel_names=CHANNEL_NAMES.tolist(),
        force=force,
    )

    return {
        "dataset_1d": str(dataset_1d_dir),
        "dataset_3d": str(dataset_3d_dir),
        "candidate_rows": int(len(candidate_manifest)),
        "folds": int(n_folds),
        "fold_output_1d": str(output_1d),
        "fold_output_3d": str(output_3d),
        "fold_rows": int(len(manifest_fold_1d)),
        "fold_tensor_shape_1d": list(tensor_fold_1d.shape),
        "fold_tensor_shape_3d": list(tensor_fold_3d.shape),
        "identical_fold_assignments": True,
        "test_evaluated": False,
    }
