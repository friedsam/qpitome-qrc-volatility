# Submission Pipeline File Inventory

## Purpose

This is the living allow-list for files used by the final Phase 3 submission pipeline. Update it whenever a required file is added, removed, renamed, transferred, or replaced.

A file belongs here only when it is required to acquire data, build and validate the canonical dataset, run a frozen model, reproduce reported results, validate a topic package, or document the judge workflow.

## Governing rules

- Scientific logic belongs under `src/`.
- Command-line files under `scripts/` remain thin adapters.
- Tests mirror the scientific source path.
- Ordinary scientific results use `results/<domain>/<experiment>/run/<run-id>/`.
- `scripts/runs/run_submission.py` is the sole judge-facing aggregation exception.
- Aggregate results are grouped by topic beneath `results/runs/<run-id>/files/`.
- Result directories may not contain copied source code.
- The fixed test partition remains unopened during development and classical reproduction.

## Canonical orchestrator and agent package

| Status | Path | Role |
|---|---|---|
| keep/update | `scripts/runs/run_submission.py` | Executes the data and `financial-classical` aggregate workflows, stops on first failure, logs commands, and hashes required outputs. |
| keep/update | `README.md` | Judge quick start and executable-scope summary. |
| keep/update | `qbraid_skill/qpitome-qrc-volatility/SKILL.md` | Binding agent procedure and safety boundary. |
| keep/update | `qbraid_skill/qpitome-qrc-volatility/references/repository-map.md` | Active path and dependency map. |
| keep/update | `qbraid_skill/qpitome-qrc-volatility/references/run-contract.md` | Canonical command, outputs, and acceptance conditions. |
| keep/update | `qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py` | Creates `.venv`, installs dependencies, runs both preflights, and executes focused tests. |
| keep | `qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py` | Existing read-only data/source preflight. |
| keep | `qbraid_skill/qpitome-qrc-volatility/scripts/preflight_classical.py` | Read-only frozen-classical dependency and parameter preflight. |
| keep/update | `tests/qbraid_skill/test_skill_contract.py` | Enforces the agent, bootstrap, path, scope, and hardware contracts. |
| keep/update | `tests/runs/test_run_submission.py` | Existing submission-runner behavior. |
| keep | `tests/runs/test_run_submission_classical.py` | Enforces command order, topic packaging, shared run ID, and absence of tuning from judge execution. |

## Transition-data dependency closure

The canonical data files and contracts remain those listed in the qBraid repository map and run contract. The classical transfer does not replace or reinterpret acquisition, cleaning, catalogue, candidate-pool, fold construction, matching, data validation, or checksum code.

Generated dataset consumed by the classical stage:

```text
results/runs/<run-id>/files/transition_forecasting/
    processed/global_transition_dataset_1d/
        cleaned_ohlc.csv.gz
        purged_walk_forward_folds/
            rematched_rolling_manifest.csv
            rematched_rolling_tensors.npz
```

## Frozen classical configuration

| Status | Path | Role |
|---|---|---|
| keep | `config/transition_forecasting/classical_benchmarks/frozen_submission.json` | Single submission parameter authority for folds, linear models, GARCH, ESN, reporting groups, and test exclusion. |

## Shared classical mechanics

| Status | Path | Role |
|---|---|---|
| keep/update | `src/baselines/garch.py` | Student-t GARCH(1,1) mechanics, explicit `arch` backend, development SciPy fallback, and log-volatility path conversion. |
| keep/update | `src/baselines/numpy_esn.py` | Deterministic ESN weights, configurable sparse connectivity, and ridge readout helpers. |
| keep | `src/baselines/esn_representation.py` | Level, first-difference, and deterministic time-coordinate transform. |
| keep | `src/evaluation/metrics.py` | Existing Phase 2 Mincer-Zarnowitz implementation. |
| keep | `src/experiments/runs.py` | Immutable scientific run-directory creation and parameter capture. |

## Classical scientific source

| Status | Path | Role |
|---|---|---|
| keep | `src/transition_forecasting/modeling/classical_benchmarks/__init__.py` | Package boundary. |
| keep | `src/transition_forecasting/modeling/classical_benchmarks/common.py` | Dataset identity checks, log-volatility QLIKE, group masks, RMSE/MZ reporting, and hashes. |
| keep | `src/transition_forecasting/modeling/classical_benchmarks/spec.py` | Frozen-spec loader and safety validation. |
| keep | `src/transition_forecasting/modeling/classical_benchmarks/linear.py` | Persistence, canonical HAR, and frozen direct sequence ridge. |
| keep | `src/transition_forecasting/modeling/classical_benchmarks/garch.py` | Causal per-origin GARCH forecasts, coverage diagnostics, and metrics. |
| keep | `src/transition_forecasting/modeling/classical_benchmarks/esn.py` | Frozen direct ESN features, pooling, readout, and shuffled control. |
| keep | `src/transition_forecasting/modeling/classical_benchmarks/esn_frozen.py` | Fixed-spec five-seed ESN producer; no tuning. |
| keep | `src/transition_forecasting/modeling/classical_benchmarks/canonical.py` | Exact-common-row model comparison, coverage, tables, and paired deltas. |
| keep | `src/transition_forecasting/modeling/classical_benchmarks/validation.py` | Complete classical topic acceptance audit. |

## Thin classical scripts

| Status | Path | Role |
|---|---|---|
| keep/thin | `scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py` | Reads frozen config and runs persistence/HAR/sequence ridge. |
| keep/thin | `scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py` | Reads frozen config and runs `arch` GARCH. |
| keep/thin | `scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py` | Reads frozen config and runs tuned direct/shuffled ESN. |
| keep/thin | `scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py` | Builds exact-common-row comparison. |
| keep/thin | `scripts/transition_forecasting/modeling/classical_benchmarks/validate_classical_run.py` | Writes `classical_baseline_audit.json` and fails on contract violations. |

## Classical tests

| Status | Path | Role |
|---|---|---|
| keep | `tests/transition_forecasting/modeling/classical_benchmarks/test_common.py` | Reporting groups, no control-by-lead rows, and exact QLIKE behavior. |
| keep | `tests/transition_forecasting/modeling/classical_benchmarks/test_esn.py` | Deterministic sparse ESN and readout shapes. |
| keep | `tests/transition_forecasting/modeling/classical_benchmarks/test_garch_mechanics.py` | Frozen GARCH identity and variance-to-log-volatility conversion. |
| keep | `tests/transition_forecasting/modeling/classical_benchmarks/test_spec.py` | Frozen folds, backend, ESN alpha, and test exclusion. |
| keep | `tests/transition_forecasting/modeling/classical_benchmarks/test_validation.py` | Exact submitted model set. |

## Standalone result mapping

```text
results/transition_forecasting/modeling/classical_benchmarks/
    linear/run/<run-id>/
    garch/run/<run-id>/
    esn/run/<run-id>/
    canonical/run/<run-id>/
```

## Aggregate topic mapping

```text
results/runs/<run-id>/files/transition_forecasting/
    validation/classical_baseline_audit.json
    modeling/classical_baselines/
        linear/run/<run-id>/
        garch/run/<run-id>/
        esn/run/<run-id>/
        canonical/run/<run-id>/
```

## Submitted classical identity

Models:

```text
persistence
har
sequence_ridge
garch_1_1_t
esn_direct_tuned
esn_shuffled_tuned
```

Groups:

```text
Transition
L1
L5
L10
Controls
Pooled
```

Metrics:

```text
RMSE
log-volatility QLIKE
Mincer-Zarnowitz alpha
Mincer-Zarnowitz beta
Mincer-Zarnowitz R2
```

The canonical comparison uses the exact finite `(fold, sample_id)` intersection across all submitted classical models. The agent workflow never invokes ESN tuning and never evaluates the fixed test partition.

## Explicit exclusions from the judge workflow

- QRC models and hardware submission;
- ESN architecture or alpha search;
- HAR-residual ESN;
- historical Stage E screens and duplicated nested scripts;
- control-by-lead reporting rows;
- development SciPy GARCH values as final submission evidence;
- source-code copies beneath results.
