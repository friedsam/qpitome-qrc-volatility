from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "transition_forecasting"
    / "modeling"
    / "run_stage_e_rate_channel_assay"
    / "run_stage_e_rate_channel_assay.py"
)
SPEC = importlib.util.spec_from_file_location("stage_e_rate_channel_assay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sequence(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)[None, :, None]


def test_relative_rate_is_causal_first_difference() -> None:
    sequence = _sequence([1.0, 1.2, 1.1, 1.5])
    rate = MODULE.build_input_family(sequence, "relative_rate")
    np.testing.assert_allclose(rate[0, :, 0], [0.0, 0.2, -0.1, 0.4])


def test_absolute_rate_differences_exponentiated_level() -> None:
    levels = np.log(np.asarray([2.0, 3.0, 2.5, 5.0]))
    rate = MODULE.build_input_family(_sequence(levels.tolist()), "absolute_rate")
    np.testing.assert_allclose(rate[0, :, 0], [0.0, 1.0, -0.5, 2.5])


def test_joint_family_preserves_level_and_rate_channels() -> None:
    sequence = _sequence([0.0, 0.1, 0.4])
    joint = MODULE.build_input_family(sequence, "level_relative_rate")
    assert joint.shape == (1, 3, 2)
    np.testing.assert_allclose(joint[0, :, 0], [0.0, 0.1, 0.4])
    np.testing.assert_allclose(joint[0, :, 1], [0.0, 0.1, 0.3])


def test_smoothed_rate_reduces_single_step_jump_amplitude() -> None:
    sequence = _sequence([0.0, 0.0, 2.0, 2.0])
    raw = MODULE.build_input_family(sequence, "relative_rate")[0, :, 0]
    smoothed = MODULE.build_input_family(sequence, "level_smoothed_relative_rate")[0, :, 1]
    assert np.max(np.abs(smoothed)) < np.max(np.abs(raw))
    np.testing.assert_allclose(smoothed, [0.0, 0.0, 1.0, 1.0])


def test_time_shuffle_preserves_multichannel_time_rows() -> None:
    sequence = np.asarray([[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]]])
    shuffled = MODULE.independently_shuffle_time(sequence, seed=7)
    assert shuffled.shape == sequence.shape
    original_rows = {tuple(row) for row in sequence[0]}
    shuffled_rows = {tuple(row) for row in shuffled[0]}
    assert shuffled_rows == original_rows
    assert all(row[1] - row[0] == 10.0 for row in shuffled[0])


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown input family"):
        MODULE.build_input_family(_sequence([0.0, 1.0]), "curvature")
