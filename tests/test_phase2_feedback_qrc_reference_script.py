from pathlib import Path

import numpy as np

from scripts.run_phase2_feedback_tfim_qrc_reference import run_reference


def test_quick_check_reference_script_path_returns_expected_outputs(tmp_path: Path):
    summary, predictions = run_reference(
        quick_check=True,
        output_dir=tmp_path,
        write_outputs=False,
    )

    expected_summary_keys = {
        "model",
        "target",
        "train_n",
        "val_n",
        "test_n",
        "n_reservoir_features",
        "train_rmse",
        "val_rmse",
        "test_rmse",
        "train_qlike",
        "val_qlike",
        "test_qlike",
        "train_mz_r2",
        "val_mz_r2",
        "test_mz_r2",
        "run_name",
    }
    assert expected_summary_keys.issubset(summary.keys())
    assert summary["model"] == "feedback_tfim_qrc_exact"
    assert summary["target"] == "future_rv_20d"
    assert summary["run_name"] == "quick_check_feedback_tfim_qrc"
    assert summary["train_n"] == 6
    assert summary["val_n"] == 3
    assert summary["test_n"] == 3
    assert summary["n_reservoir_features"] > 0

    expected_prediction_columns = {
        "split",
        "date",
        "actual_future_rv_20d",
        "qrc_pred_future_rv_20d",
        "run_name",
    }
    assert expected_prediction_columns.issubset(predictions.columns)
    assert len(predictions) == 12
    assert set(predictions["split"]) == {"train", "val", "test"}
    assert np.isfinite(predictions["actual_future_rv_20d"]).all()
    assert np.isfinite(predictions["qrc_pred_future_rv_20d"]).all()
    assert (predictions["actual_future_rv_20d"] > 0).all()
    assert (predictions["qrc_pred_future_rv_20d"] > 0).all()
    assert predictions["run_name"].eq("quick_check_feedback_tfim_qrc").all()

    assert not list(tmp_path.iterdir())
