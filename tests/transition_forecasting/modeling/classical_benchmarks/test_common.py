from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.classical_benchmarks.common import (
    REQUIRED_GROUPS,
    TARGET_COLUMNS,
    group_masks,
    load_rematched_dataset,
    metric_row,
    qlike_loss,
)


def test_groups_do_not_create_control_leads():
    frame = pd.DataFrame(
        {"label": [1, 1, 1, 0, 0], "lead": [1, 5, 10, 1, 5]}
    )
    masks = group_masks(frame)
    assert tuple(masks) == REQUIRED_GROUPS
    assert masks["Controls"].sum() == 2
    assert masks["L1"].sum() == 1
    assert not (masks["L1"] & masks["Controls"]).any()


def test_qlike_zero_for_exact_log_volatility_forecast():
    observed = np.array([[-2.0, -1.5]])
    assert np.allclose(qlike_loss(observed, observed), 0.0)
    row = metric_row(
        model="x",
        group="Pooled",
        y_true=observed,
        y_pred=observed,
    )
    assert row["rmse"] == 0.0
    assert abs(row["qlike"]) < 1e-12


def test_loader_accepts_active_rematched_manifest_without_legacy_stratum(
    tmp_path: Path,
) -> None:
    fold_root = tmp_path / "purged_walk_forward_folds"
    fold_root.mkdir(parents=True)
    rows = []
    for index, (sample_id, label, lead) in enumerate(
        (("p1", 1, 1), ("p5", 1, 5), ("p10", 1, 10), ("n1", 0, 1))
    ):
        row = {
            "sample_id": sample_id,
            "episode_id": "episode",
            "index": "^GSPC",
            "origin_date": f"2020-01-{index + 1:02d}",
            "event_onset": "2020-01-15",
            "label": label,
            "lead": lead,
            "fold": 4,
            "fold_split": "val",
            "market_group": "US",
            "level": 1.0,
            "mean5": 1.0,
            "mean20": 1.0,
            "matched_positive_id": "p1" if label == 0 else None,
            "match_distance": 0.1 if label == 0 else None,
        }
        row.update({column: 1.0 for column in TARGET_COLUMNS})
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(fold_root / "rematched_rolling_manifest.csv", index=False)
    np.savez_compressed(
        fold_root / "rematched_rolling_tensors.npz",
        X=np.zeros((len(manifest), 40, 1), dtype=float),
        sample_id=manifest["sample_id"].astype(str).to_numpy(),
        fold=manifest["fold"].astype(int).to_numpy(),
        fold_split=manifest["fold_split"].astype(str).to_numpy(),
    )

    loaded = load_rematched_dataset(tmp_path)

    assert "control_stratum" in loaded.manifest.columns
    assert loaded.manifest["control_stratum"].isna().all()
    assert loaded.manifest["matched_positive_id"].notna().sum() == 1
