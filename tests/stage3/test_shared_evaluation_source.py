import importlib


def test_shared_evaluation_modules_import() -> None:
    modules = [
        "qpitome_qrc.evaluation.binary",
        "qpitome_qrc.evaluation.historical_crossfit",
        "qpitome_qrc.evaluation.residualization",
        "qpitome_qrc.evaluation.scoring",
        "qpitome_qrc.evaluation.metrics",
        "qpitome_qrc.baselines.logistic_offset",
        "qpitome_qrc.baselines.fixed_esn",
        "qpitome_qrc.baselines.priors",
    ]
    for name in modules:
        assert importlib.import_module(name) is not None


def test_public_evaluation_api_exposes_core_helpers() -> None:
    evaluation = importlib.import_module("qpitome_qrc.evaluation")
    for name in [
        "logistic_pipeline",
        "residualize_train_test",
        "residualize_train_test_safe",
    ]:
        assert callable(getattr(evaluation, name))
