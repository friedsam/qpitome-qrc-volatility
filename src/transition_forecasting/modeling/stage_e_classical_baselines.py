"""Compatibility exports for the canonical Case151 QRC reproduction.

The active submission owns the scientific definitions in
``transition_forecasting.modeling.classical_benchmarks.common``. The historical
QRC modules keep their original import path without duplicating those definitions.
"""
from transition_forecasting.modeling.classical_benchmarks.common import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    qlike_loss,
)

__all__ = ["HAR_FEATURES", "TARGET_COLUMNS", "qlike_loss"]
