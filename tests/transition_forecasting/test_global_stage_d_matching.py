from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.transition_events import LEADS, NEG_PER_POS
from transition_forecasting.modeling.global_stage_d_dataset import (
    _validate_manifest,
    build_global_stage_d_dataset,
)


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dates = pd.bdate_range("2009-01-01", periods=900)
    trend = np.linspace(100.0, 180.0, len(dates))
    wave = 2.0 * np.sin(np.arange(len(dates)) / 13.0)
    close = trend + wave
    open_ = close * (1.0 + 0.001 * np.sin(np.arange(len(dates)) / 7.0))
    high = np.maximum(open_, close) * 1.012
    low = np.minimum(open_, close) * 0.988

    ohlc = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    ohlc_path = tmp_path / "SYNTH.csv"
    ohlc.to_csv(ohlc_path, index=False)

    onset = dates[500]
    catalogue = pd.DataFrame(
        {
            "index": ["SYNTH"],
            "market_group": ["synthetic"],
            "episode_id": ["E0001"],
            "onset_date": [onset],
            "effective_start": [dates[0]],
        }
    )
    catalogue_path = tmp_path / "representative_transition_catalogue.csv"
    catalogue.to_csv(catalogue_path, index=False)

    inventory = pd.DataFrame({"index": ["SYNTH"], "path": [str(ohlc_path)]})
    inventory_path = tmp_path / "global_index_ohlc_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    return catalogue_path, inventory_path


def test_stage_d_builds_exact_control_ratio_without_reuse(tmp_path: Path) -> None:
    catalogue_path, inventory_path = _write_synthetic_inputs(tmp_path)
    manifest, tensor, _, summary = build_global_stage_d_dataset(
        catalogue_path,
        inventory_path,
    )

    positives = manifest[manifest["label"] == 1]
    controls = manifest[manifest["label"] == 0]

    assert len(positives) == len(LEADS)
    assert len(controls) == len(LEADS) * NEG_PER_POS
    assert summary["controls_per_positive_target"] == NEG_PER_POS
    assert summary["positive_samples"] == len(positives)
    assert summary["negative_samples"] == len(controls)
    assert tensor.shape[0] == len(manifest)

    counts = controls.groupby("matched_positive_id").size()
    assert set(counts.index) == set(positives["sample_id"])
    assert counts.eq(NEG_PER_POS).all()

    for (_, lead, split), group in controls.groupby(["index", "lead", "split"]):
        assert not group["origin_pos"].duplicated().any(), (lead, split)


def test_stage_d_matching_is_deterministic(tmp_path: Path) -> None:
    catalogue_path, inventory_path = _write_synthetic_inputs(tmp_path)

    first_manifest, first_tensor, _, _ = build_global_stage_d_dataset(
        catalogue_path,
        inventory_path,
    )
    second_manifest, second_tensor, _, _ = build_global_stage_d_dataset(
        catalogue_path,
        inventory_path,
    )

    pd.testing.assert_frame_equal(
        first_manifest.reset_index(drop=True),
        second_manifest.reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )
    np.testing.assert_array_equal(first_tensor, second_tensor)


def test_manifest_validation_rejects_incomplete_controls() -> None:
    rows = [
        {
            "sample_id": "P_E0001_SYNTH_L1",
            "label": 1,
            "episode_id": "E0001",
            "split": "train",
            "matched_positive_id": None,
        }
    ]
    rows.extend(
        {
            "sample_id": f"N_P_E0001_SYNTH_L1_{i}",
            "label": 0,
            "episode_id": "E0001",
            "split": "train",
            "matched_positive_id": "P_E0001_SYNTH_L1",
        }
        for i in range(1, NEG_PER_POS)
    )

    with np.testing.assert_raises_regex(ValueError, "without exactly"):
        _validate_manifest(pd.DataFrame(rows))


def test_manifest_validation_rejects_episode_split_leakage() -> None:
    rows = [
        {
            "sample_id": "P_E0001_SYNTH_L1",
            "label": 1,
            "episode_id": "E0001",
            "split": "train",
            "matched_positive_id": None,
        }
    ]
    rows.extend(
        {
            "sample_id": f"N_P_E0001_SYNTH_L1_{i}",
            "label": 0,
            "episode_id": "E0001",
            "split": "val" if i == 1 else "train",
            "matched_positive_id": "P_E0001_SYNTH_L1",
        }
        for i in range(1, NEG_PER_POS + 1)
    )

    with np.testing.assert_raises_regex(ValueError, "span multiple splits"):
        _validate_manifest(pd.DataFrame(rows))
