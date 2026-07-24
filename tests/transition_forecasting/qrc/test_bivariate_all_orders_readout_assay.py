from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from transition_forecasting.qrc.bivariate_all_orders_readout_assay import (
    BivariateAllOrdersReadoutConfig,
    build_all_orders_representations,
    materialize_source_run,
)


def _write_completed_source(root: Path) -> Path:
    run = root / "bivariate_bilinear_mixing_001"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "bivariate_bilinear_mixing_complete",
                "config": {
                    "samples": 160,
                    "sequence_length": 20,
                    "memory_delays": [1, 2, 4, 5, 8, 12],
                    "alphas": [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0],
                    "seeds": [20260724],
                    "step_duration_us": 0.02,
                    "interaction_scale": 1.25,
                    "drive_phase_rad": 0.0,
                    "permutations": 4,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run


def test_all_orders_representations_have_expected_widths() -> None:
    rng = np.random.default_rng(20260724)
    raw = {
        "palindrome": rng.normal(size=(32, 63)),
        "d1_then_x2": rng.normal(size=(32, 63)),
        "d2_then_x1": rng.normal(size=(32, 63)),
        "x2_then_d1": rng.normal(size=(32, 63)),
        "x1_then_d2": rng.normal(size=(32, 63)),
    }

    representations = build_all_orders_representations(raw)

    assert representations["palindrome_control"].shape == (32, 63)
    assert representations["all_orders_joint"].shape == (32, 252)
    assert representations["palindrome_plus_all_orders"].shape == (32, 315)
    np.testing.assert_allclose(
        representations["all_orders_joint"][:, :63], raw["d1_then_x2"]
    )
    np.testing.assert_allclose(
        representations["palindrome_plus_all_orders"][:, :63], raw["palindrome"]
    )


def test_source_run_directory_is_resolved(tmp_path: Path) -> None:
    expected = _write_completed_source(tmp_path / "nested")

    with materialize_source_run(tmp_path) as resolved:
        assert resolved == expected


def test_source_run_zip_is_resolved(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _write_completed_source(source)
    archive_path = tmp_path / "bilinear.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(
            expected / "summary.json",
            arcname=f"{expected.name}/summary.json",
        )

    with materialize_source_run(archive_path) as resolved:
        assert resolved.name == expected.name
        assert json.loads((resolved / "summary.json").read_text())["status"] == (
            "bivariate_bilinear_mixing_complete"
        )


def test_source_run_rejects_ambiguous_completed_runs(tmp_path: Path) -> None:
    _write_completed_source(tmp_path / "first")
    _write_completed_source(tmp_path / "second")

    with pytest.raises(ValueError, match="exactly one"):
        with materialize_source_run(tmp_path):
            pass


def test_configuration_requires_positive_permutations() -> None:
    BivariateAllOrdersReadoutConfig(permutations=None).validate()
    BivariateAllOrdersReadoutConfig(permutations=4).validate()
    with pytest.raises(ValueError, match="positive"):
        BivariateAllOrdersReadoutConfig(permutations=0).validate()
