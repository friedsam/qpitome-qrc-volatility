# Classical Transition Benchmark Work Manifest

**Status:** locally implemented and tested; pending deliberate port to the active submission branch.

## Scientific contract

- Dataset: `global_transition_dataset_1d.zip`, active `purged_walk_forward_folds` lineage only.
- Selection folds: 4–6.
- Frozen confirmation folds: 7–8.
- Test rows used: 0.
- Reported groups: Transition, L1, L5, L10, Controls, Pooled. Controls are never subdivided by lead.
- Metrics: existing Stage E log-volatility QLIKE, RMSE, and existing Phase 2 Mincer–Zarnowitz intercept/slope/R².
- Models: persistence, canonical HAR, direct sequence ridge, Student-t GARCH(1,1), direct ESN, shuffled ESN control. No QRC and no residual ESN.

## Files to port

| Path | Action | Provenance |
|---|---|---|
| `docs/transition_forecasting/modeling/classical_benchmarks/README.md` | add | Reproduction commands and output contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/__init__.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/common.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/linear.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/garch.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/esn.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/esn_selection_worker.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/esn_fold_worker.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/esn_finalize.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/transition_forecasting/modeling/classical_benchmarks/canonical.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py` | add | Thin CLI runner. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py` | add | Thin CLI runner. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py` | add | Resumable thin CLI runner. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py` | add | Thin CLI runner. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_common.py` | add | Metric and population contract. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_linear.py` | add | Opt-in current-data integration test. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_garch.py` | add | Opt-in current-data integration test. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_garch_mechanics.py` | add | GARCH mechanics test. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_esn.py` | add | ESN equivalence tests. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_canonical.py` | add | Opt-in current-run integration test. |
| `src/baselines/garch.py` | update | Preserve existing `arch` backend; add deterministic SciPy fallback for environments without optional `arch`. |
| `pyproject.toml` | update | Declare SciPy explicitly for the GARCH fallback. |

## Verified current-data runs

- Linear: `results/transition_forecasting/modeling/classical_benchmarks/linear/linear_current_data_001`.
- GARCH: `results/transition_forecasting/modeling/classical_benchmarks/garch/garch_current_data_001`.
- ESN: `results/transition_forecasting/modeling/classical_benchmarks/esn/esn_current_data_001`.
- Canonical: `results/transition_forecasting/modeling/classical_benchmarks/canonical/classical_current_data_002`.
- File hashes are captured in each run's `dataset_manifest.json` and in the packaged delivery manifest.

## Validation

- With `QPITOME_CLASSICAL_DATASET_ROOT` and `QPITOME_CLASSICAL_RESULTS_ROOT` set to the verified current-data artifacts, `python -m pytest -q`: **8 passed**.
- Without local current-data artifacts, the portable unit suite passes and three opt-in integration tests are skipped.
- `python -m compileall -q src scripts`: passed.
- Direct ESN CLI `--help` and `finalize` phase: passed.
- All prediction outputs contain validation rows only; test rows used = 0.

## Known limitations

- The current GARCH run used the SciPy fallback because `arch` was not installed in the execution environment. The model specification remains zero-mean Student-t GARCH(1,1), but fitted values may differ from an `arch` backend run.
- The direct ESN is intentionally resumable in three phases because a monolithic process retained large BLAS workspaces in this runtime. The numerical specification and fixed-fold predictions are unchanged.
- These files must be ported deliberately to the latest submission tree; this work branch must not be merged wholesale into an unrelated active branch.
