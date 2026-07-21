from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import (
    best_alpha_per_condition,
    feature_names_from_metadata,
    make_prediction_frame,
    signal_diagnostics,
    write_feature_archive,
    write_transition_overlays,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [1, 1],
            "sample_id": ["a", "b"],
            "fold_split": ["train", "val"],
            "lead": [1, 1],
            "label": [1, 1],
            "episode_id": ["ea", "eb"],
            "origin_date": ["2020-01-01", "2020-02-01"],
        }
    )


def test_feature_names_match_emitted_order() -> None:
    metadata = {
        "probe_steps": [2, 4],
        "nearest_pairs": [[0, 1], [1, 2]],
        "long_pairs": [[0, 2]],
        "positions_um": [[0.0, 0.0], [8.0, 0.0], [17.0, 0.0]],
        "feature_count": 22,
    }
    names = feature_names_from_metadata(metadata)
    assert len(names) == 22
    assert names[0] == "probe_2_occupation_site_0"
    assert names[-1] == "probe_4_long_connected_0_2"


def test_prediction_frame_and_signal_diagnostics() -> None:
    frame = _frame()
    y = np.asarray([[1.0, 2.0], [2.0, 4.0]])
    prediction = np.asarray([[1.0, 2.5], [1.5, 3.5]])
    output = make_prediction_frame(
        frame,
        y,
        prediction,
        fold=1,
        condition="ordered",
        alpha=100.0,
        readout_mode="direct",
    )
    assert len(output) == 4
    assert output["horizon"].tolist() == [1, 2, 1, 2]
    diagnostics = signal_diagnostics(y, prediction)
    assert diagnostics["prediction_std"] > 0.0
    assert np.isfinite(diagnostics["correlation"])


def test_feature_archive_is_pickle_free(tmp_path) -> None:
    frame = _frame()
    block = {
        "condition": "ordered",
        "frame": frame,
        "features": np.arange(8, dtype=float).reshape(2, 4),
        "targets": np.ones((2, 2)),
        "har_predictions": np.zeros((2, 2)),
        "source_level": np.ones((2, 3)),
        "encoded_level": np.ones((2, 3)),
    }
    path = tmp_path / "features.npz"
    write_feature_archive(
        path,
        feature_blocks=[block],
        feature_names=("f0", "f1", "f2", "f3"),
    )
    with np.load(path, allow_pickle=False) as data:
        assert data["feature_matrix"].shape == (2, 4)
        assert data["feature_names"].tolist() == ["f0", "f1", "f2", "f3"]
        assert data["condition"].tolist() == ["ordered", "ordered"]


def test_best_alpha_and_overlay_outputs(tmp_path) -> None:
    metrics = pd.DataFrame(
        {
            "fold": [1, 2, 1, 2],
            "condition": ["ordered"] * 4,
            "alpha": [10.0, 10.0, 100.0, 100.0],
            "val_qlike": [2.0, 2.0, 1.0, 1.2],
            "val_rmse": [1.0, 1.0, 0.8, 0.9],
        }
    )
    best = best_alpha_per_condition(metrics)
    assert best.iloc[0]["alpha"] == 100.0

    frame = pd.DataFrame(
        {
            "fold": [1],
            "sample_id": ["sample"],
            "fold_split": ["val"],
            "lead": [1],
            "label": [1],
            "episode_id": ["event"],
            "origin_date": ["2020-01-01"],
        }
    )
    y = np.asarray([[0.2, 0.4]])
    predictions = [
        make_prediction_frame(
            frame,
            y,
            y,
            fold=1,
            condition="har_only",
            alpha=None,
            readout_mode="baseline",
        ),
        make_prediction_frame(
            frame,
            y,
            y * 0.9,
            fold=1,
            condition="ordered",
            alpha=100.0,
            readout_mode="direct",
        ),
    ]
    windows = pd.DataFrame(
        {
            "fold": [1],
            "sample_id": ["sample"],
            "fold_split": ["val"],
            "lead": [1],
            "label": [1],
            "episode_id": ["event"],
            "origin_date": ["2020-01-01"],
            "level_window": [np.asarray([-0.2, -0.1, 0.0])],
        }
    )
    files = write_transition_overlays(
        tmp_path,
        predictions=pd.concat(predictions, ignore_index=True),
        window_records=windows,
        best_alphas=best,
    )
    assert "transition_mean_overlay_lead_1.png" in files
    assert (tmp_path / "transition_mean_overlay_lead_1.png").is_file()
    assert (tmp_path / "transition_peak_event_overlay_lead_1.png").is_file()
