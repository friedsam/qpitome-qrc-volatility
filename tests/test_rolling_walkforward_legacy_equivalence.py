from tests.legacy_rolling_walkforward_reference import legacy_make_windows

from qpitome_qrc.evaluation.walkforward import make_purged_rolling_windows


def strip_purge(windows):
    return [
        {
            "window": w["window"],
            "train": w["train"],
            "val": w["val"],
            "test": w["test"],
        }
        for w in windows
    ]


def test_historical_annual_geometry_matches_frozen_oracle():
    kwargs = dict(
        min_train=2500,
        val_size=504,
        purge=60,
        test_size=252,
        step=252,
    )

    for n_rows in (5000, 7000, 7658, 8500):
        expected = legacy_make_windows(n_rows, **kwargs)
        actual = make_purged_rolling_windows(n_rows, **kwargs)

        assert strip_purge(actual) == expected


def test_overlapping_rolling_geometry_matches_frozen_oracle():
    kwargs = dict(
        min_train=1000,
        val_size=250,
        purge=20,
        test_size=200,
        step=100,
    )

    expected = legacy_make_windows(3000, **kwargs)
    actual = make_purged_rolling_windows(3000, **kwargs)

    assert strip_purge(actual) == expected


def test_shared_geometry_records_explicit_purge_interval():
    windows = make_purged_rolling_windows(
        5000,
        min_train=2500,
        val_size=504,
        purge=60,
        test_size=252,
        step=252,
    )

    first = windows[0]

    assert first["train"] == (0, 2500)
    assert first["val"] == (2500, 3004)
    assert first["purge"] == (3004, 3064)
    assert first["test"] == (3064, 3316)


def test_incomplete_trailing_window_is_discarded():
    windows = make_purged_rolling_windows(
        3400,
        min_train=2500,
        val_size=504,
        purge=60,
        test_size=252,
        step=252,
    )

    assert len(windows) == 1
    assert windows[0]["test"] == (3064, 3316)
