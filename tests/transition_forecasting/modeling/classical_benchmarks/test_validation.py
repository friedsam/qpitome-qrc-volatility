from transition_forecasting.modeling.classical_benchmarks.validation import MODELS


def test_submission_model_set_is_only_agreed_classical_models():
    assert MODELS == (
        "persistence",
        "har",
        "sequence_ridge",
        "garch_1_1_t",
        "esn_direct_tuned",
        "esn_shuffled_tuned",
    )
