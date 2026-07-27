"""Compatibility exports for legacy Stage-E classical modules.

The active submission implementation owns shared target-column definitions in
``transition_forecasting.modeling.classical_benchmarks.common``. Historical
modules may continue importing the former Stage-E path without duplicating the
scientific contract.
"""

from transition_forecasting.modeling.classical_benchmarks.common import TARGET_COLUMNS

__all__ = ["TARGET_COLUMNS"]
