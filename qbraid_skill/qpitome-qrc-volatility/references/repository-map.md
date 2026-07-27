# Repository Map for the qBraid Agent

This reference describes the active `main` dependency closure. It is a navigation aid, not permission to reinterpret scientific behavior.

## Judge-facing entry point

The Agent executes exactly one orchestration helper:

```text
qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py
```

This helper owns only sequencing, literal run-ID generation, path construction, stage validation, failure propagation, and `agent_scope_manifest.json`. It contains no scientific model implementation and no hardware command.

It delegates to the established canonical entry points:

```text
scripts/runs/run_submission_stage_layout.py financial-classical
scripts/reproduction/run_case151_simulation.py --verification-mode current-pipeline --archive-existing-failed
scripts/runs/run_submission_benchmarks.py all --profile smoke
scripts/runs/run_submission_benchmarks.py validate-existing --profile smoke
```

The default Agent scope is `full-smoke`. Explicit `core` stops after current-pipeline Case151 verification.

## Governing layout

Normal scientific components preserve:

```text
src/<domain>/<experiment>/*.py
scripts/<domain>/<experiment>/*.py
tests/<domain>/<experiment>/test_*.py
results/<domain>/<experiment>/run/<run_id>/*
```

Judge-facing aggregation is the intentional exception:

```text
results/runs/<RUN_ID>/
    run_manifest.json
    agent_scope_manifest.json
    logs/
    files/
        data/
        classical_baselines/
        qrc/
            simulation/run/<RUN_ID>/
            simulation/run/failed_attempts/
        quantum_studies/
        mnist/
```

Run directories may contain data products, parameters, manifests, logs, predictions, metrics, figures, checksums, and validation reports. They must not contain copied source code.

## Data and classical stages

Stage-layout wrapper:

```text
scripts/runs/run_submission_stage_layout.py
```

Delegates to:

```text
scripts/runs/run_submission.py
scripts/runs/run_submission_layout.py
```

Canonical workflow:

```text
financial-classical
```

It includes verified transition-data construction, persistence/HAR/sequence ridge, Student-t GARCH(1,1), frozen direct and shuffled ESNs, exact-common-row tables, and validation.

Data scripts:

```text
scripts/transition_forecasting/data/acquire_global_index_data.py
scripts/transition_forecasting/data/build_transition_datasets.py
scripts/transition_forecasting/data/build_transition_folds.py
scripts/transition_forecasting/data/validate_transition_run.py
scripts/transition_forecasting/data/freeze_transition_checksums.py
```

Classical parameter authority:

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
```

Classical model scripts:

```text
scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py
scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py
scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py
scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py
scripts/transition_forecasting/modeling/classical_benchmarks/validate_classical_run.py
```

Reported models:

```text
persistence
har
sequence_ridge
garch_1_1_t
esn_direct_tuned
esn_shuffled_tuned
```

Reported groups:

```text
Transition
L1
L5
L10
Controls
Pooled
```

## Canonical Case151 QRC

Runner:

```text
scripts/reproduction/run_case151_simulation.py
```

Frozen authority:

```text
config/case151/expected_metrics.json
config/case151/agent_run_spec.json
reference/case151/freeze_001/
```

Reusable source includes:

```text
src/transition_forecasting/qrc/palindrome_real_task_relevance_assay.py
src/transition_forecasting/qrc/bivariate_crossover_assay.py
src/transition_forecasting/qrc/bivariate_capacity_dynamics.py
src/transition_forecasting/qrc/temporal_rydberg_chain.py
src/transition_forecasting/qrc/temporal_rydberg_ladder.py
src/transition_forecasting/qrc/representation_candidates.py
src/transition_forecasting/qrc/ladder_finite_shot_sampling.py
```

Canonical identity:

```text
six atoms
A/4 -> B/2 -> A/4 palindrome
three probe times
occupation_pair_raw
63 features
fold-specific chronological readout selection
fold-8 alpha 0.1
fold-8 lambda 0.25
intercept-free residual head
zero financial test rows
```

Verification modes:

```text
historical-oracle
    exact metric equality on the historical fold lineage from canonical commit
    40ec805cc2b4efe416c0a57f1c599cca6def92c3 and source run
    palindrome_real_task_002

current-pipeline
    same frozen model identity on newly generated financial-classical folds;
    immutable historical reference hashes are verified and current metric deltas
    are reported without claiming exact historical reproduction
```

A failed or incomplete aggregate Case151 attempt is moved beneath `files/qrc/simulation/run/failed_attempts/` before retry. A verified output is never overwritten.

## Phase-3 smoke families

Agent-level benchmark runner:

```text
scripts/runs/run_submission_benchmarks.py
```

Scientific entry points:

```text
scripts/transition_forecasting/data/acquire_mnist.py
scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py
scripts/transition_forecasting/qrc/run_palindrome_noise_assay.py
scripts/transition_forecasting/qrc/run_palindrome_scaling_assay.py
scripts/transition_forecasting/qrc/run_palindrome_shot_assay.py
```

The `all --profile smoke` pass runs bounded MNIST, noise, scaling, and finite-shot studies. The subsequent `validate-existing --profile smoke` pass verifies required outputs and writes the final artifact inventory without recomputation.

## Hardware boundary

Retrieval-only collector:

```text
scripts/hardware/aquila_case151_collect.py
```

It is outside `core` and `full-smoke`. The Agent orchestration helper never imports or invokes it. No hardware query, retrieval, selection, packaging, or submission occurs during reproduction.

## Focused tests

```text
tests/qbraid_skill/test_skill_contract.py
tests/runs/test_run_submission.py
tests/runs/test_run_submission_classical.py
tests/runs/test_run_submission_benchmarks.py
tests/transition_forecasting/data/test_fold_datasets.py
tests/transition_forecasting/modeling/classical_benchmarks/
tests/transition_forecasting/qrc/test_case151_reproduction_contract.py
tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py
tests/transition_forecasting/qrc/test_palindrome_rydberg_noise.py
tests/transition_forecasting/qrc/test_palindrome_scaling_assay.py
tests/transition_forecasting/qrc/test_palindrome_shot_assay.py
```

## Scientific identity

- one input channel: `log_volatility_level`;
- 40 trading sessions per input sequence;
- 10 trading sessions per target path;
- transition leads 1, 5, and 10;
- three matched controls per retained positive;
- eight purged walk-forward folds;
- folds 4–6 are model-selection history;
- folds 7–8 are confirmation recheck;
- fixed financial test chronology remains unopened;
- classical execution uses frozen parameters and does not retune;
- canonical Case151 uses the frozen exact 63-feature simulator;
- historical Case151 metrics remain tied to their historical fold lineage;
- current-pipeline Case151 metrics are reported separately with explicit deltas;
- full-smoke is bounded integration coverage, not a primary-result replacement.
