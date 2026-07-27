"""Spatial ladder mode weights required by the canonical feature extractor."""
from __future__ import annotations

import numpy as np


def ladder_mode_weights() -> np.ndarray:
    row_symmetric = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    row_antisymmetric = np.asarray([1.0, -1.0]) / np.sqrt(2.0)
    column_constant = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    column_gradient = np.asarray([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    column_curvature = np.asarray([1.0, -2.0, 1.0]) / np.sqrt(6.0)
    weights = [
        np.outer(row, column).reshape(-1)
        for row in (row_symmetric, row_antisymmetric)
        for column in (column_constant, column_gradient, column_curvature)
    ]
    matrix = np.column_stack(weights)
    if not np.allclose(matrix.T @ matrix, np.eye(6), atol=1e-12, rtol=0.0):
        raise RuntimeError("ladder spatial mode weights are not orthonormal")
    return matrix
