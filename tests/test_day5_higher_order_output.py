"""Tests for exact-state higher-order output blocks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_higher_order_output_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_higher_order", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def binary_bits(n_atoms: int) -> np.ndarray:
    values = np.arange(2**n_atoms, dtype=np.uint16)
    return ((values[:, None] >> np.arange(n_atoms)) & 1).astype(float)


def test_exact_output_block_shapes_and_histogram_normalization() -> None:
    n_atoms = 10
    bits = binary_bits(n_atoms)
    rng = np.random.default_rng(2)
    states = rng.normal(size=(4, 2**n_atoms)) + 1j * rng.normal(size=(4, 2**n_atoms))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    blocks = module.exact_output_blocks(states, bits)
    assert blocks["count_histogram"].shape == (4, 10)
    assert blocks["blockade_patterns"].shape == (4, 6)
    assert blocks["pooled_triples"].shape == (4, 3)
    assert blocks["higher_order_all"].shape == (4, 19)
    assert np.all(blocks["count_histogram"].sum(axis=1) <= 1.0 + 1e-12)


def test_zero_state_has_expected_collective_statistics() -> None:
    bits = binary_bits(10)
    states = np.zeros((1, 2**10), dtype=complex)
    states[0, 0] = 1.0
    blocks = module.exact_output_blocks(states, bits)
    hist = blocks["count_histogram"][0]
    blockade = blocks["blockade_patterns"][0]
    triples = blocks["pooled_triples"][0]
    assert hist[0] == 1.0
    assert np.all(hist[1:] == 0.0)
    assert blockade[0] == 1.0
    assert np.all(blockade[1:] == 0.0)
    assert np.allclose(triples, 0.0)


def test_exact_state_evolution_is_normalized() -> None:
    config = module.base.base.assay.rydberg_config()
    patterns = np.full((3, 10), 0.5)
    states = module.evolve_local_detuning_states(patterns, config)
    assert states.shape == (3, 2**10)
    assert np.allclose(np.sum(np.abs(states) ** 2, axis=1), 1.0, atol=1e-10)
