"""Frozen pre-refactor rolling-window geometry.

This intentionally duplicates the historical implementation from
run_rolling_har_esn_context.py and must not import production protocol code.
"""


def legacy_make_windows(
    n,
    min_train,
    val_size,
    purge,
    test_size,
    step,
):
    first_test = min_train + val_size + purge
    rows = []
    start = first_test
    wid = 1

    while start + test_size <= n:
        val_end = start - purge
        val_start = val_end - val_size
        rows.append(
            {
                "window": wid,
                "train": (0, val_start),
                "val": (val_start, val_end),
                "test": (start, start + test_size),
            }
        )
        start += step
        wid += 1

    return rows
