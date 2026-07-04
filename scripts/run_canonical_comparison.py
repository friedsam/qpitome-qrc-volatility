#!/usr/bin/env python3
"""Safe entry point for the canonical Phase 3 master comparison.

This thin entry point loads ``run_master_comparison.py`` and applies three
runtime corrections identified during static review of the initial master
implementation:

1. JSON manifest keys from pandas groupby tuples are converted to strings.
2. Finite-shot Rydberg feature generation uses separate deterministic RNG
   streams for train, validation, and test splits.
3. The default feature cache is redirected to ignored ``scratch/`` storage so
   expensive intermediate arrays are not accidentally committed.

Use this file for canonical runs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


MASTER_PATH = Path(__file__).with_name("run_master_comparison.py")


def load_master():
    spec = importlib.util.spec_from_file_location("phase3_master_comparison", MASTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MASTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def ensure_safe_cache_default() -> None:
    if "--cache-dir" not in sys.argv:
        sys.argv.extend(["--cache-dir", "scratch/canonical_cache"])


def main() -> None:
    ensure_safe_cache_default()
    master = load_master()

    original_dumps = master.json.dumps

    def safe_dumps(value, *args, **kwargs):
        return original_dumps(json_safe(value), *args, **kwargs)

    master.json.dumps = safe_dumps

    def split_seed_feature_block(*, cache_dir, cache_payload, arrays, config, force):
        key = master.cache_key(cache_payload)
        path = cache_dir / f"rydberg_{key}.npz"
        if path.exists() and not force:
            loaded = np.load(path)
            return {split: loaded[split] for split in master.SPLITS}, 0.0, str(path)

        start = time.perf_counter()
        block = {}
        base_seed = 0 if config.shot_seed is None else int(config.shot_seed)
        for split_index, split in enumerate(master.SPLITS):
            use_config = config
            if config.shots is not None:
                use_config = replace(config, shot_seed=base_seed + 1_000 * split_index)
            block[split] = master.build_rydberg_feature_matrix(arrays[split], use_config)
        elapsed = time.perf_counter() - start

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **block)
        return block, elapsed, str(path)

    master.load_or_build_feature_block = split_seed_feature_block
    master.main()


if __name__ == "__main__":
    main()
