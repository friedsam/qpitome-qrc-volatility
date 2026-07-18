from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/analyze_stage_e_esn_features.py")
SPEC = importlib.util.spec_from_file_location("stage_e_esn_feature_assay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest() -> pd.DataFrame:
    rows = []
    sample = 0
    for split, count in (("train", 20), ("val", 8), ("test", 8)):
        for local in range(count):
            sample += 1
            level = -4.0 + 0.02 * sample
            mean5 = level + 0.01
            mean20 = level - 0.02
            rows.append({
                "sample_id": f"S{sample}",
                "episode_id": f"E_{split}_{local}",
                "label": local % 2,
                "lead": (1, 5, 10)[local % 3],
                "split": split,
                "level": level,
                "mean5": mean5,
                "mean20": mean20,
                "slope5": 0.001 * local,
                "slope20": 0.0005 * local,
                "std20": 0.1 + 0.001 * local,
                "max20": level + 0.1,
                **{f"target_x_h{h + 1}": level + 0.01 * (h + 1) for h in range(10)},
            })
    return pd.DataFrame(rows)


def test_feature_assay_returns_all_tables() -> None:
    manifest = _manifest()
    rng = np.random.default_rng(3)
    states = rng.normal(size=(len(manifest), 12))
    states[:, 0] += manifest["level"].to_numpy() * 2.0
    metadata = {"name": "tiny", "n": 12, "seed": 1}

    result = MODULE.analyze_state_file(
        states,
        manifest,
        metadata,
        pls_components=(1, 2),
        ridge_alphas=(1.0, 10.0),
    )

    assert set(result) == {"pca", "coordinates", "reconstruction", "pls"}
    assert not result["pca"].empty
    assert len(result["coordinates"]) == 12
    assert {"level", "mean5", "mean20"}.issubset(set(result["reconstruction"]["target"]))
    assert set(result["pls"]["n_components"]) == {1, 2}
    assert np.isfinite(result["pls"][["train_qlike", "val_qlike", "val_rmse"]]).all().all()


def test_load_state_file_supports_object_string_metadata(tmp_path: Path) -> None:
    manifest = _manifest()
    states = np.arange(len(manifest) * 5, dtype=float).reshape(len(manifest), 5)
    cache_path = tmp_path / "reservoir_states__tiny__seed1.npz"
    config = {"name": "tiny", "n": 4, "sr": 0.9, "inp": 0.3, "leak": 0.3}

    np.savez_compressed(
        cache_path,
        states=states,
        sample_id=manifest["sample_id"].astype(object).to_numpy(),
        config_json=np.asarray(json.dumps(config), dtype=object),
        seed=np.asarray(1),
    )

    loaded_states, metadata = MODULE._load_state_file(cache_path, manifest)

    assert np.array_equal(loaded_states, states)
    assert metadata["name"] == "tiny"
    assert metadata["n"] == 4
    assert metadata["seed"] == 1
