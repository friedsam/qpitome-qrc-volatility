"""Compatibility exports for migrated Stage-E and QRC modules.

The active submission implementation owns the shared HAR feature, target-column,
and QLIKE definitions in
``transition_forecasting.modeling.classical_benchmarks.common``. Historical QRC
modules may continue importing the former Stage-E path without creating a
second classical implementation or metric source of truth.
"""

from transition_forecasting.modeling.classical_benchmarks.common import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    qlike_loss,
)

__all__ = ["HAR_FEATURES", "TARGET_COLUMNS", "qlike_loss"]
