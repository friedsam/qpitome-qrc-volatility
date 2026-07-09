"""Corrected entry point for HAR-residual input-family quickshot.

The original runner computed causal HAR forecasts only in sparse episode
neighborhoods. That is sufficient for local windows but leaves NaN gaps that a
continuous-state ESN cannot cross. This wrapper preserves the experiment and
replaces only HAR-aware channel construction with one contiguous causal span.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).with_name("run_branch_har_residual_input_families.py")
spec = importlib.util.spec_from_file_location("branch_har_residual_input_families", RUNNER)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load {RUNNER}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def add_har_aware_channels_contiguous(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Build HAR-aware channels over one contiguous causal daily span."""
    out = mod.add_base_ratios(daily)

    earliest_branch = int(episodes["branch_idx"].astype(int).min())

    # Earliest local path row needs forecasts lagged by 20 rows and differenced
    # by 5 rows. Compute from there through the dataset end so the generic
    # continuous-state ESN sees one fully contiguous finite sequence.
    first_needed = max(
        0,
        earliest_branch - mod.LOOKBACK + 1 - 25,
    )
    needed_rows = np.arange(first_needed, len(out), dtype=int)

    har_pred = mod.causal_har_predictions_for_rows(out, needed_rows)
    out["causal_har_pred_daily"] = har_pred
    out["har_log_future_to_current_daily"] = np.log(
        out["causal_har_pred_daily"] / out["rv_20d"]
    )
    out["har_forecast_revision_5d"] = np.log(
        out["causal_har_pred_daily"] / out["causal_har_pred_daily"].shift(5)
    )
    out["har_observable_log_error_20d"] = np.log(
        out["rv_20d"] / out["causal_har_pred_daily"].shift(20)
    )
    out["har_observable_error_change_5d"] = (
        out["har_observable_log_error_20d"].diff(5)
    )

    bounds = {
        "har_log_future_to_current_daily": (-2.0, 2.0),
        "har_forecast_revision_5d": (-1.5, 1.5),
        "har_observable_log_error_20d": (-2.0, 2.0),
        "har_observable_error_change_5d": (-2.0, 2.0),
        "log_rv5_over_rv20": (-2.0, 2.0),
        "log_rv20_over_rv60": (-1.5, 1.5),
        "rv5_change_scaled": (-3.0, 3.0),
    }
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].clip(lo, hi)
    return out


mod.add_har_aware_channels = add_har_aware_channels_contiguous

if __name__ == "__main__":
    mod.main()
