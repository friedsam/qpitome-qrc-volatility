from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.stage_e_classical_baselines import StageEData

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_esn_regularization_audit.py")
SPEC = importlib.util.spec_from_file_location("stage_e_esn_regularization", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _data() -> StageEData:
    rng = np.random.default_rng(9)
    rows = []
    sequences = []
    sample = 0
    for split, count in (("train", 12), ("val", 6), ("test", 6)):
        for local in range(count):
            sample += 1
            level = -4.5 + 0.025 * sample
            sequence = level + np.linspace(-0.08, 0.0, 40) + rng.normal(0.0, 0.004, 40)
            target = level + np.linspace(0.01, 0.10, 10)
            rows.append({
                "sample_id": f"S{sample}",
                "episode_id": f"E_{split}_{local}",
                "label": local % 2,
                "lead": (1, 5, 10)[local % 3],
                "split": split,
                "level": level,
                "mean5": float(sequence[-5:].mean()),
                "mean20": float(sequence[-20:].mean()),
                **{f"target_x_h{h + 1}": float(target[h]) for h in range(10)},
            })
            sequences.append(sequence[:, None])
    return StageEData(pd.DataFrame(rows), np.asarray(sequences))


def test_regularization_audit_crosses_alpha_and_formulation(tmp_path: Path) -> None:
    config = {"name": "tiny", "n": 8, "sr": 0.7, "inp": 0.2, "leak": 0.3}
    audit = MODULE.run_regularization_audit(
        _data(),
        configs=(config,),
        alphas=(0.1, 10.0),
        seeds=(1,),
        har_alpha=1.0,
        reservoir_cache_dir=tmp_path,
    )
    assert len(audit) == 4
    assert set(audit["formulation"]) == {"direct", "har_residual"}
    assert set(audit["alpha"]) == {0.1, 10.0}
    assert set(audit["state_source"]) == {"built"}
    assert np.isfinite(audit[["train_qlike", "val_qlike", "coefficient_norm"]]).all().all()

    cache_path = tmp_path / "reservoir_states__tiny__seed1.npz"
    assert cache_path.exists()
    with np.load(cache_path, allow_pickle=False) as cached:
        assert cached["states"].shape == (24, 9)
        assert cached["sample_id"].astype(str).tolist()[0] == "S1"
        assert cached["split"].astype(str).tolist()[-1] == "test"

    repeated = MODULE.run_regularization_audit(
        _data(),
        configs=(config,),
        alphas=(0.1,),
        seeds=(1,),
        har_alpha=1.0,
        reservoir_cache_dir=tmp_path,
    )
    assert set(repeated["state_source"]) == {"loaded"}

    summary = MODULE.summarize(audit)
    assert summary["test_evaluated"] is False
    assert len(summary["best_seed_averaged_rows"]) == 2
