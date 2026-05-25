"""
tests/test_imports.py
---------------------
Verifies that all main project modules import without errors.
No real data or GPU required.
"""


def test_import_loader():
    import src.data.loader  # noqa: F401


def test_import_preprocessor():
    import src.data.preprocessor  # noqa: F401


def test_import_cross_val():
    import src.evaluation.cross_val  # noqa: F401


def test_import_extractor():
    import src.features.extractor  # noqa: F401


def test_import_logistic_regression():
    import src.models.logistic_regression  # noqa: F401


def test_import_random_forest():
    import src.models.random_forest  # noqa: F401


def test_import_cnn1d():
    import src.models.cnn1d  # noqa: F401


def test_import_metrics():
    import src.evaluation.metrics  # noqa: F401


def test_key_constants_exposed():
    from src.features.extractor import TOTAL_FEATURES, N_FEATURES_PER_CHANNEL, N_CHANNELS
    assert TOTAL_FEATURES == N_FEATURES_PER_CHANNEL * N_CHANNELS

    from src.evaluation.cross_val import N_FOLDS
    assert N_FOLDS == 5

    from src.data.preprocessor import SAMPLING_RATE_HZ
    assert SAMPLING_RATE_HZ == 100.0
