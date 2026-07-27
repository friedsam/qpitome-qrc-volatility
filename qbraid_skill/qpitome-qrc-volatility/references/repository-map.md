# Repository Map for the qBraid Agent

This reference describes the active `stage1-dev` dependency closure. It is a navigation aid, not permission to reinterpret scientific behavior.

## Governing layout

Normal scientific components preserve:

```text
src/<domain>/<experiment>/*.py
scripts/<domain>/<experiment>/*.py
tests/<domain>/<experiment>/test_*.py
results/<domain>/<experiment>/run/<run_id>/*
```

`scripts/runs/run_submission_stage_layout.py` is the sole judge-facing aggregation exception. It delegates scientific execution to the established runner and collects topic packages under:

```text
results/runs/<run-id>/files/
```

Run directories may contain data products, parameters, manifests, logs, predictions, metrics, figures, checksums, and validation reports. They must not contain copied source code.

## Canonical entry point

```text
scripts/runs/run_submission_stage_layout.py
```

The established execution engine remains:

```text
scripts/runs/run_submission.py
```

The stage-layout helper is:

```text
scripts/runs/run_submission_layout.py
```

Active workflows:

```text
data
validate-data
transition-data
financial-classical
```

`financial-classical` contains the complete `transition-data` command plan followed by the frozen classical producers and validator.

## Aggregate topic structure

```text
results/runs/<run-id>/
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
            linear/run/<run-id>/
            garch/run/<run-id>/
            esn/run/<run-id>/
            canonical/run/<run-id>/
            validation/
                classical_baseline_audit.json
        qrc/
        quantum_studies/
        mnist/
```

Only stages included in the selected workflow are created. Completed hardware evidence belongs under `qrc/hardware/` and is not rerun by the default agent workflow.

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
src/transition_forecasting/modeling/classical_benchmarks/common.py
src/transition_forecasting/modeling/classical_benchmarks/spec.py
src/transition_forecasting/modeling/classical_benchmarks/linear.py
src/transition_forecasting/modeling/classical_benchmarks/garch.py
src/transition_forecasting/modeling/classical_benchmarks/esn.py
src/transition_forecasting/modeling/classical_benchmarks/esn_frozen.py
src/transition_forecasting/modeling/classical_benchmarks/canonical.py
src/transition_forecasting/modeling/classical_benchmarks/validation.py
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

## Focused tests

```text
tests/qbraid_skill/test_skill_contract.py
tests/runs/test_run_submission.py
tests/runs/test_run_submission_classical.py
tests/transition_forecasting/data/test_fold_datasets.py
tests/transition_forecasting/modeling/test_stage_d_candidate_pool.py
tests/transition_forecasting/modeling/test_chronological_control_matching.py
tests/transition_forecasting/modeling/test_chronological_rematched_dataset.py
tests/transition_forecasting/modeling/test_chronological_splits.py
tests/transition_forecasting/modeling/classical_benchmarks/test_common.py
tests/transition_forecasting/modeling/classical_benchmarks/test_esn.py
tests/transition_forecasting/modeling/classical_benchmarks/test_garch_mechanics.py
tests/transition_forecasting/modeling/classical_benchmarks/test_spec.py
tests/transition_forecasting/modeling/classical_benchmarks/test_validation.py
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
- classical submission execution uses frozen parameters and does not retune.

Historical and exploratory modules may remain importable. Their presence does not make them canonical. Determine the active path from `scripts/runs/run_submission_stage_layout.py` and its direct imports.
