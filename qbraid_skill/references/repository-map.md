# Repository Map for the qBraid Agent

This reference describes the current `stage1-dev` dependency closure. It is a navigation aid, not permission to reinterpret scientific behavior.

## Governing layout

Normal scientific components should preserve a human-readable mirror:

```text
src/<domain>/<experiment>/*.py
scripts/<domain>/<experiment>/*.py
tests/<domain>/<experiment>/test_*.py
results/<domain>/<experiment>/run/<run_id>/*
```

Shared reusable modules may live above an experiment-specific directory when several experiments consume them.

`scripts/runs/run_submission.py` is the sole planned result-layout exception. It is the automated judge-facing orchestrator and may collect many paper-critical artifacts beneath one structured aggregate run. That exception must not become the default for ordinary experiment producers.

Run directories contain data products, parameters, manifests, logs, predictions, metrics, figures, and checksums. They must not contain source-code copies.

## Current Stage-1 entry point

```text
scripts/runs/run_submission.py
```

Current workflow:

```text
transition-data
```

Current aggregate output:

```text
results/runs/<run-id>/
```

This aggregate layout is provisional until the final judge-facing taxonomy is frozen.

## Thin Stage-1 scripts

```text
scripts/transition_forecasting/data/acquire_global_index_data.py
scripts/transition_forecasting/data/build_transition_datasets.py
scripts/transition_forecasting/data/build_transition_folds.py
scripts/transition_forecasting/data/validate_transition_run.py
scripts/transition_forecasting/data/freeze_transition_checksums.py
```

These scripts are command-line adapters. Scientific behavior belongs in reusable modules under `src/`.

## Reusable transition-data source

```text
src/transition_forecasting/catalogue/
src/transition_forecasting/data/
src/transition_forecasting/modeling/
src/transition_forecasting/quality/
```

Important modules:

```text
src/transition_forecasting/catalogue/global_transition_catalogue.py
src/transition_forecasting/data/acquisition.py
src/transition_forecasting/data/cleaning.py
src/transition_forecasting/data/dataset.py
src/transition_forecasting/data/fold_datasets.py
src/transition_forecasting/data/modeling_inputs.py
src/transition_forecasting/data/validation.py
src/transition_forecasting/data/volatility.py
src/transition_forecasting/modeling/stage_d_candidate_pool.py
src/transition_forecasting/modeling/chronological_splits.py
src/transition_forecasting/modeling/chronological_control_matching.py
src/transition_forecasting/modeling/chronological_rematched_dataset.py
src/transition_forecasting/quality/structural_bad_prints.py
```

## Frozen contracts

```text
config/transition_forecasting/contracts/global_index_ohlc_inventory.csv
config/transition_forecasting/contracts/global_range_quality.csv
```

Treat these as scientific contracts. Do not update them during reproduction.

## Focused tests

```text
tests/qbraid_skill/test_skill_contract.py
tests/transition_forecasting/data/test_fold_datasets.py
tests/transition_forecasting/modeling/test_stage_d_candidate_pool.py
tests/transition_forecasting/modeling/test_chronological_control_matching.py
tests/transition_forecasting/modeling/test_chronological_rematched_dataset.py
tests/transition_forecasting/modeling/test_chronological_splits.py
tests/runs/test_run_submission.py
```

## Current scientific identity

The Stage-1 implementation uses:

- one input channel: `log_volatility_level`;
- 40 trading sessions per input sequence;
- 10 trading sessions per target path;
- forecast leads 1, 5, and 10;
- binary transition/control labels;
- three controls per retained positive;
- eight purged walk-forward folds;
- a 17% newest fixed-test chronology;
- exact trading-row input/target intervals;
- fold-local control rematching;
- no control-origin reuse within a fold partition, including across leads;
- an unopened fixed test set during development.

Do not revive the retired three-channel representation or abandoned redesigned-control fields when working on this branch.

## Historical and noncanonical code

Some modules retain historical helpers for provenance or compatibility. Their presence does not make them canonical. Determine the active path from the current runner and direct imports; do not invoke a plausible-looking legacy function merely because it remains importable.

In particular, do not substitute the older end-to-end helpers in `transition_events.py` for the active global-catalogue and fold pipeline.