from transition_forecasting.data.fold_datasets import DEFAULT_N_FOLDS


def test_default_fold_count_is_eight() -> None:
    assert DEFAULT_N_FOLDS == 8
