# Repository Map for the qBraid Agent

This reference describes the active `main` dependency closure. It is a navigation aid, not permission to reinterpret scientific behavior.

## Governing layout

Normal scientific components preserve:

```text
src/<domain>/<experiment>/*.py
scripts/<domain>/<experiment>/*.py
tests/<domain>/<experiment>/test_*.py
results/<domain>/<experiment>/run/<run_id>/*
```

Judge-facing aggregation is the intentional exception. All reproduced outputs are collected beneath:

```text
results/runs/<RUN_ID>/files/
```

Run directories may contain data products, parameters, manifests, logs, predictions, metrics, figures, checksums, and validation reports. They must not contain copied source code.

## Canonical Agent sequence

### Data and classical stages

```text
scripts/runs/run_submission_stage_layout.py
```

This delegates to:

```text
scripts/runs/run_submission.py
scripts/runs/run_submission_layout.py
```

The required workflow is:

```text
financial-classical
```

It contains the complete transition-data plan followed by the frozen classical producers and validator.

### Canonical Case151 QRC stage

```text
scripts/reproduction/run_case151_simulation.py
```

It consumes the fold tensors created by `financial-classical`, writes beneath `files/qrc/simulation/run/<RUN_ID>/`, and verifies the frozen Case151 metric and identity oracle.

### Optional Phase-3 smoke stages

```text
scripts/runs/run_submission_benchmarks.py
```

The `all --profile smoke` workflow runs bounded MNIST, noise, scaling, and finite-shot studies. `validate-existing --profile smoke` verifies their required artifacts without recomputation.

### Hardware

```text
scripts/hardware/aquila_case151_collect.py
```

This is retrieval-only and outside the default Agent workflow. The Agent must not query, retrieve, select, or submit hardware during core or full-smoke reproduction.

## Aggregate topic structure

```text
results/runs/<RUN_ID>/
    run_manifest.json
    logs/
    files/
        data/
            raw/
            processed/global_transition_dataset_1d/
            validation/
                data_pipeline_audit.json
                data_pipeline_checksums.json
        classical_baselines/
            linear/run/<RUN_ID>/
            garch/run/<RUN_ID>/
            esn/run/<RUN_ID>/
            canonical/run/<RUN_ID>/
            validation/
                classical_baseline_audit.json
        qrc/
            simulation/run/<RUN_ID>/
            hardware/<HARDWARE_RUN_ID>/
        quantum_studies/
            benchmark_manifest.json
            benchmark_artifact_inventory.json
            noise/run/<RUN_ID>/
            scaling/run/<RUN_ID>/
            shots/run/<RUN_ID>/
        mnist/
            raw/
            shards/
            run/<RUN_ID>/
```

Only stages included in the requested scope are created.

## Transition-data source

Thin scripts:

```text
scripts/transition_forecasting/data/acquire_global_index_data.py
scripts/transition_forecasting/data/build_transition_datasets.py
scripts/transition_forecasting/data/build_transition_folds.py
scripts/transition_forecasting/data/validate_transition_run.py
scripts/transition_forecasting/data/freeze_transition_checksums.py
```

Reusable source:

```text
src/transition_forecasting/catalogue/
src/transition_forecasting/data/
src/transition_forecasting/modeling/chronological_control_matching.py
src/transition_forecasting/modeling/chronological_rematched_dataset.py
src/transition_forecasting/modeling/chronological_splits.py
src/transition_forecasting/modeling/stage_d_candidate_pool.py
src/transition_forecasting/quality/
```

Frozen data contracts:

```text
config/transition_forecasting/contracts/global_index_ohlc_inventory.csv
config/transition_forecasting/contracts/global_range_quality.csv
```

## Classical comparison

Frozen parameter authority:

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
```

Reusable source:

```text
src/transition_forecasting/modeling/classical_benchmarks/
src/baselines/garch.py
src/baselines/numpy_esn.py
src/baselines/esn_representation.py
src/evaluation/metrics.py
src/experiments/runs.py
```

Thin scripts:

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

The canonical comparison restricts all models to the exact intersection of finite `(fold, sample_id)` predictions before calculating RMSE, log-volatility QLIKE, and Mincer-Zarnowitz diagnostics.

## Canonical Case151 QRC

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
zero test rows
```

## Optional Phase-3 benchmark families

```text
scripts/transition_forecasting/data/acquire_mnist.py
scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py
scripts/transition_forecasting/qrc/run_palindrome_noise_assay.py
scripts/transition_forecasting/qrc/run_palindrome_scaling_assay.py
scripts/transition_forecasting/qrc/run_palindrome_shot_assay.py
```

The Agent-level benchmark runner owns their shared aggregate paths, logs, manifests, required-output inventory, resume behavior, and read-only validation.

## Focused tests

```text
tests/qbraid_skill/test_skill_contract.py
tests/runs/test_run_submission.py
tests/runs/test_run_submission_classical.py
tests/runs/test_run_submission_benchmarks.py
tests/transition_forecasting/data/test_fold_datasets.py
tests/transition_forecasting/modeling/test_stage_d_candidate_pool.py
tests/transition_forecasting/modeling/test_chronological_control_matching.py
tests/transition_forecasting/modeling/test_chronological_rematched_dataset.py
tests/transition_forecasting/modeling/test_chronological_splits.py
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
- fixed test chronology remains unopened;
- classical submission execution uses frozen parameters and does not retune;
- canonical Case151 uses the frozen exact 63-feature simulator and oracle;
- full-smoke is bounded integration coverage, not a primary-result replacement.
