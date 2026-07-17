from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "results"
    / "transition_destination"
    / "frozen_reproduction"
    / "imported_phase3_refactor"
)


def test_frozen_path_bundle_is_internally_consistent() -> None:
    paths = np.load(RESULTS / "paths.npy")
    metadata = pd.read_csv(RESULTS / "episode_metadata.csv")
    standardization = pd.read_csv(RESULTS / "standardization.csv")

    assert paths.ndim == 3
    assert paths.shape[0] == len(metadata)
    assert paths.shape[1] == 13
    assert paths.shape[2] == len(standardization)
    assert set(metadata["y_positive"].unique()) <= {0, 1}
    assert metadata["outcome_available_date"].notna().all()


def test_frozen_holdout_and_reference_metrics() -> None:
    metrics = pd.read_csv(RESULTS / "classical_metrics.csv")
    all_rows = metrics[metrics["subgroup"] == "all"].set_index("model")

    assert int(all_rows.loc["momentum_13w", "n_predictions"]) == 48
    assert int(all_rows.loc["momentum_13w", "n_positive"]) == 35
    assert int(all_rows.loc["momentum_13w", "n_negative"]) == 13
    assert np.isclose(
        all_rows.loc["momentum_13w", "log_loss"],
        0.5510822250050053,
        rtol=0.0,
        atol=1e-12,
    )

    rydberg = pd.read_csv(RESULTS / "rydberg_metrics.csv").set_index("model")
    assert np.isclose(
        rydberg.loc[
            "momentum_plus_rydberg_return_uncertainty", "roc_auc"
        ],
        0.6813186813186813,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        rydberg.loc[
            "momentum_plus_rydberg_return_uncertainty", "log_loss"
        ],
        0.5591484943152198,
        rtol=0.0,
        atol=1e-12,
    )


def test_outcome_maturity_is_respected_in_frozen_predictions() -> None:
    predictions = pd.read_csv(
        RESULTS / "classical_predictions.csv",
        parse_dates=["date", "latest_training_outcome_date"],
    )
    assert (
        predictions["latest_training_outcome_date"] < predictions["date"]
    ).all()
