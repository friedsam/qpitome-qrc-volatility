from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from transition_forecasting.data.validation import sha256_file


CHANNEL_NAMES = np.asarray(
    ["log_volatility_level", "relative_rate", "absolute_rate"],
    dtype="U32",
)


def build_three_channel_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Derive a three-channel dataset without changing sample identity or order.

    Channels are constructed causally from each stored 40-row one-channel sequence:
    1. log-volatility level;
    2. first difference of log-volatility (relative rate);
    3. first difference after exponentiating log-volatility (absolute rate).

    The first rate value is zero because the observation immediately preceding the
    stored window is not available.
    """
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    source_npz = source_dir / "sequence_tensors.npz"
    if not source_npz.is_file():
        raise FileNotFoundError(f"Missing source tensor archive: {source_npz}")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
        shutil.rmtree(output_dir)

    with np.load(source_npz, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "X" not in arrays:
        raise KeyError("sequence_tensors.npz does not contain X")
    x = np.asarray(arrays["X"])
    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError(f"Expected one-channel tensor shaped (n, time, 1), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Source tensor contains non-finite values")

    level = x[:, :, 0]
    relative_rate = np.diff(level, axis=1, prepend=level[:, :1])
    volatility = np.exp(np.clip(level, -20.0, 20.0))
    absolute_rate = np.diff(volatility, axis=1, prepend=volatility[:, :1])
    x3 = np.stack((level, relative_rate, absolute_rate), axis=-1)
    if not np.isfinite(x3).all():
        raise ValueError("Derived three-channel tensor contains non-finite values")

    output_dir.mkdir(parents=True, exist_ok=False)
    for path in source_dir.iterdir():
        if path.name in {"sequence_tensors.npz", "manifest.json"}:
            continue
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)

    arrays["X"] = x3
    arrays["channel_names"] = CHANNEL_NAMES
    np.savez_compressed(output_dir / "sequence_tensors.npz", **arrays)

    source_manifest_path = source_dir / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"] = "global_transition_dataset_3d"
    manifest["derived_from"] = {
        "path": str(source_dir),
        "manifest_sha256": sha256_file(source_manifest_path),
        "sequence_tensors_sha256": sha256_file(source_npz),
    }
    manifest["representation"] = {
        "shape": list(x3.shape),
        "dtype": str(x3.dtype),
        "channels": CHANNEL_NAMES.tolist(),
        "rate_initialization": "zero at first stored row",
        "sample_identity_and_order_preserved": True,
        "level_channel_bitwise_identical_to_1d": True,
    }
    manifest["test_evaluated"] = False
    manifest.pop("files", None)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "shape": list(x3.shape),
        "dtype": str(x3.dtype),
        "channels": CHANNEL_NAMES.tolist(),
        "samples_preserved": int(x3.shape[0]),
        "test_evaluated": False,
    }
