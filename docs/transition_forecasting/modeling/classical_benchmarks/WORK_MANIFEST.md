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
| `src/transition_forecasting/modeling/classical_benchmarks/esn_tuning.py` | add | Bounded transition-first direct-ESN tuning over previously established dynamical regimes with explicit baseline and temporal-order guardrails. |
| `src/transition_forecasting/modeling/classical_benchmarks/canonical.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_esn_tuning.py` | add | Thin CLI for the bounded transition-first direct-ESN tuning run. |
| `scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_common.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_linear.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_garch.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_garch_mechanics.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_esn.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_esn_tuning.py` | add | Verifies transition-first ranking, baseline guardrails, and ordered-over-shuffled admissibility. |
| `tests/transition_forecasting/modeling/classical_benchmarks/test_canonical.py` | add | Adapts the existing Stage E benchmark lineage to the current rematched dataset and submission reporting contract. |
| `src/baselines/garch.py` | update | Preserve existing `arch` backend; add deterministic SciPy fallback for environments without optional `arch`. |
| `pyproject.toml` | update | Declare SciPy explicitly for the GARCH fallback. |

## Verified current-data runs

### linear

- Result directory: `results/transition_forecasting/modeling/classical_benchmarks/linear/linear_current_data_001`
- `summary.json` SHA-256: `1c43c690e0d5fc66bb71540eb4a6577844990fac3c414f16add93e62807885c1`
- `submission_metrics.csv` SHA-256: `0c44dc0fe986d5f1b0c151841065ff9c385b9ced5c8d04cb28425f8ffc8e8a8b`
- `dataset_manifest.json` SHA-256: `9a57c33169e1f103866f42cdf759bc59a14cba0517a28c4b724b7c77e1f9b0e4`
- `runtime.json` SHA-256: `597ce679b301bf4aad8d27762b7672f089045c4dca8d90b4ab5138ea76f4b329`

### garch

- Result directory: `results/transition_forecasting/modeling/classical_benchmarks/garch/garch_current_data_001`
- `summary.json` SHA-256: `12f0612d9d1e67cbbcf64bedd393ef0d6d5101330a98a2c72955853eaf6e8290`
- `submission_metrics.csv` SHA-256: `393918905841a8f135f7d2cd65a652eda40b3455b34ca25cbce08363fc713620`
- `dataset_manifest.json` SHA-256: `4d856838024417aac674694c14ab98470870429e852e34ee291f6a7e773cdbe3`
- `runtime.json` SHA-256: `6d199cea7968607efb3355b179c455896e2986dbb995d39ec2009117088faf03`

### esn

- Result directory: `results/transition_forecasting/modeling/classical_benchmarks/esn/esn_current_data_001`
- `summary.json` SHA-256: `a8835fc9eb18fcd81b1be66eb2a8edb1907d76990d40c05395ff913e79df4ddc`
- `submission_metrics.csv` SHA-256: `e476ab258d24c172fd77503bd2c4486fc2771bacfe5bb62b30a84c539bd21658`
- `dataset_manifest.json` SHA-256: `20283d91c764b3e06863de81a3022d2adf7dc32334718ad63889034099d09e02`
- `runtime.json` SHA-256: `c1b4c30558f38baab0f8c60da617dae9553063cd4b481c252b0e02101c536a3a`

### esn tuning

- Result directory: `results/transition_forecasting/modeling/classical_benchmarks/esn_tuning/esn_transition_tuned_001`
- Selected specification: `short_leak095`, 300 units, spectral radius 0.55, input scale 0.20, leak 0.95, connectivity 0.02, alpha 120000, seeds 1–5.
- `summary.json` SHA-256: `bcb321be9c9cdd4d459bdf2b2f48ab966a4a9211fe38c6fb1cdbbfeadb74d971`
- `submission_metrics.csv` SHA-256: `96e16fa88798d24fd8502031b165c96a86ec627c512e4b254d8dd02dcb48a0a5`
- `predictions.csv.gz` SHA-256: `a940cb889f35307451bf5fb30dd8f6122ce50730a1a68d057101095c7ab3fadb`
- Test rows used: 0.

### canonical

- Result directory: `results/transition_forecasting/modeling/classical_benchmarks/canonical/classical_current_data_002`
- `summary.json` SHA-256: `2786c3b60777fe7ebcc009b1525310ec7a60ce53bf0e41eccf0c721bfa3e68c0`
- `submission_metrics.csv` SHA-256: `6a3c96d79a94b832f1d23776907c216f06cd96c12a6171c960c1a0ba4beb8132`
- `runtime.json` SHA-256: `b6882ad772554fbaadf64278fdfdbc5b350927929453237f2d81d82ba3294998`

## Validation

- With `QPITOME_CLASSICAL_DATASET_ROOT` and `QPITOME_CLASSICAL_RESULTS_ROOT` set to the verified current-data artifacts, `python -m pytest -q`: **8 passed**.
- Without local current-data artifacts, the portable unit suite passes and three opt-in integration tests are skipped.
- `python -m compileall -q src scripts`: passed.
- Direct ESN CLI `--help` and `finalize` phase: passed.
- All prediction outputs contain validation rows only; test rows used = 0.
- ESN tuning unit/equivalence suite: **3 passed**; source and runner compilation passed.

## Known limitations

- The current GARCH run used the SciPy fallback because `arch` was not installed in the execution environment. The model specification remains zero-mean Student-t GARCH(1,1), but fitted values may differ from an `arch` backend run.
- The direct ESN is intentionally resumable in three phases because a monolithic process retained large BLAS workspaces in this runtime. The numerical specification and fixed-fold predictions are unchanged.
- The transition-first tuning specification was selected only on folds 4–6, but folds 7–8 had already been viewed during earlier bounded tuning iterations. The stored folds 7–8 result is therefore a confirmation recheck, not a pristine one-shot holdout.
- These files must be ported deliberately to the latest submission tree; this work branch must not be merged wholesale into an unrelated active branch.
