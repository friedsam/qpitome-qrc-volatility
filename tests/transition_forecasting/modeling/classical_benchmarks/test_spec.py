from transition_forecasting.modeling.classical_benchmarks.spec import load_frozen_spec


def test_frozen_spec_is_submission_safe():
    spec = load_frozen_spec()
    assert spec["selection_folds"] == [4, 5, 6]
    assert spec["confirmation_folds"] == [7, 8]
    assert spec["garch"]["backend"] == "arch"
    assert spec["esn"]["alpha"] == 120000.0
    assert spec["reporting"]["test_evaluated"] is False
