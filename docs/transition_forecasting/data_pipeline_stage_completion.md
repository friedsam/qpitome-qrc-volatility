# Transition-Forecasting Data Pipeline: Stage Completion Record

## Purpose

This document records the final design, implementation, validation, and submission-port requirements for the transition-forecasting data pipeline. It is intended to preserve the reasoning needed for the Phase 3 writeup and to prevent later reconstruction of the pipeline from scattered code and development logs.

## Scientific objective

The pipeline constructs a leakage-controlled dataset for forecasting volatility-regime transitions across global equity indices. Each positive sample represents a known transition event and each control is selected from a full eligible candidate pool using fold-local matching. Evaluation uses purged chronological folds and leaves the newest fixed test partition untouched until final evaluation.

## Canonical datasets

### One-channel representation

The canonical one-channel dataset contains the pre-origin log-volatility level:

```text
data/processed/transition_forecasting/global_transition_dataset_1d
```

Validated properties:

- 286,015 cleaned OHLC rows;
- 232,630 daily volatility rows;
- 383 clustered transition events;
- 4,596 canonical samples;
- 1,149 positives;
- 3,447 matched controls;
- 35 indices;
- 198 global episodes;
- 3 controls per positive;
- sequence tensor shape `(4596, 40, 1)`;
- finite `float64` tensor values;
- fixed test partition not evaluated during development.

### Three-channel representation

The deterministic three-channel dataset is stored at:

```text
data/processed/transition_forecasting/global_transition_dataset_3d
```

Channels:

1. `log_volatility_level`;
2. first difference of log volatility, with the initial value set to zero;
3. normalized sequence time from zero to one.

Validated properties:

- tensor shape `(4596, 40, 3)`;
- exact deterministic transform of the one-channel tensor;
- identical sample identifiers and ordering;
- finite `float64` values.

The Rydberg implementation is expected to use level and difference as physical inputs, with time represented implicitly by sequential evolution. The explicit time channel remains useful for classical controls and representation studies.

## Candidate-pool construction

The control candidate pool is rebuilt directly from the frozen daily-volatility series and transition catalogue rather than copied from historical development artifacts.

Outputs are written into each dataset directory:

```text
control_candidate_manifest.csv
control_candidate_tensors.npz
candidate_pool_summary.json
```

Validated candidate count:

```text
566,898 rows
```

The one-channel candidate tensor is transformed deterministically into the three-channel candidate tensor. Candidate identities, index membership, lead time, and row order remain aligned across representations.

## Chronological evaluation design

The fold methodology preserves the previously validated Stage E logic:

- reserve the newest 17% of the chronology as a fixed test partition;
- construct three expanding train/next-validation folds;
- represent each sample interval as input start through forecast target end;
- apply a 10-day embargo;
- purge same-index samples whose full intervals overlap validation or test intervals;
- enforce strict chronology;
- perform fold-, split-, index-, and lead-local control rematching;
- use deterministic nearest-neighbor selection on the frozen matching features;
- prevent origin reuse within a partition, including across forecast leads;
- require exactly three controls per complete positive sample;
- leave the fixed test partition unevaluated during model development.

## Fold artifacts

One-channel folds:

```text
data/processed/transition_forecasting/global_transition_dataset_1d/
    purged_walk_forward_folds/
```

Three-channel folds:

```text
data/processed/transition_forecasting/global_transition_dataset_3d/
    purged_walk_forward_folds/
```

Each directory contains:

```text
rematched_rolling_manifest.csv
rematched_rolling_tensors.npz
control_match_audit.csv
summary.json
```

Validated fold results:

- 10,272 fold rows;
- one-channel tensor shape `(10272, 40, 1)`;
- three-channel tensor shape `(10272, 40, 3)`;
- identical fold assignments across representations;
- identical selected samples and sample ordering;
- deterministic three-channel transform of every one-channel fold tensor;
- fixed test partition not evaluated.

## Historical parity validation

The historical corrected fold entry point was rerun independently against the canonical one-channel dataset. The organized pipeline and historical pipeline produced:

- identical tensor values;
- identical sample identifiers;
- identical sample ordering;
- identical fold assignments;
- identical selected controls;
- identical manifest and audit structure.

The only numerical differences were machine-precision-scale variations in stored matching distances and statistics derived from them, generally around `1e-15`. These differences did not alter ranking, selection, membership, or tensors.

Parity conclusion:

```text
PASS
```

## Pickle-free serialization

String metadata in candidate and fold NPZ files is stored as fixed-width Unicode rather than NumPy object arrays.

Verified with `allow_pickle=False`:

```text
1D candidate pool
  X: float64
  candidate_id: <U26
  index: <U14
  lead: int64

3D candidate pool
  X: float64
  candidate_id: <U26
  index: <U14
  lead: int64

1D folds
  X: float64
  sample_id: <U30
  fold: int64
  fold_split: <U5
  channel_names: <U32

3D folds
  X: float64
  sample_id: <U30
  fold: int64
  fold_split: <U5
  channel_names: <U32
```

No fold or candidate artifact requires pickle-enabled loading.

## Current reusable implementation

Core reusable files:

```text
src/transition_forecasting/data/three_channel.py
src/transition_forecasting/data/fold_datasets.py
src/transition_forecasting/modeling/chronological_splits.py
src/transition_forecasting/modeling/chronological_control_matching.py
src/transition_forecasting/modeling/chronological_rematched_dataset.py
src/transition_forecasting/modeling/stage_d_candidate_pool.py
```

Current development entry points:

```text
scripts/development_runs/build_processed_dataset.py
scripts/development_runs/validate_processed_dataset.py
scripts/development_runs/build_purged_walk_forward_folds.py
```

Historical scripts are retained only as audit references and should not be ported wholesale into the final submission branch.

## Submission-run design

The final runner owns the run identifier and creates:

```text
results/runs/<run-id>/
```

Reusable scientific functions should receive explicit input and output paths. They should not know about `run_id` or infer global result locations.

The submission runner will derive run-local paths and pass them to thin scripts. This allows a complete judge rerun to remain internally consistent while keeping the scientific functions reusable.

The final submission run may write generated datasets, folds, metrics, predictions, plots, logs, and manifests beneath the same run directory. Existing SPY/VIX workflows must remain available and must not be replaced by the transition-forecasting pipeline.

## Porting policy

After a stage is validated, its exact dependency closure should be ported selectively to `stage1-dev` so another machine can test the submission branch while later stages are still under development.

Port or merge:

- reusable source modules used by the final runner;
- thin submission entry points;
- direct tests for the selected modules;
- required configuration;
- runner changes;
- stage documentation.

Do not port wholesale:

- historical GPT-2 scripts;
- obsolete Stage A/B/D/E orchestration;
- development-only parameter sweeps;
- exploratory cross-market diagnostics;
- generated development results;
- parity-reconstruction outputs.

## Remaining work after this stage

The data pipeline itself is complete. Remaining submission work includes:

1. adapt all completed data functions and scripts to explicit run-local output paths;
2. extend the submission runner without removing SPY/VIX;
3. selectively port the completed data stage to `stage1-dev`;
4. verify a clean-environment rerun on a second Mac;
5. implement the classical transition baselines;
6. finalize the Rydberg reservoir experiments;
7. implement and launch the common MNIST benchmark sufficiently early for reruns;
8. aggregate final metrics and figures;
9. incorporate this record into the Phase 3 narrative and final report.

## Writeup points to preserve

The final report should emphasize:

- the task is causal forecasting, not retrospective transition classification;
- controls are rematched inside each chronological partition;
- interval purging includes input and forecast-target windows;
- the newest test partition remains untouched during development;
- the three-channel representation is a deterministic extension of the canonical one-channel dataset;
- one-channel folds reproduce the historical corrected pipeline exactly at the sample and tensor level;
- machine-precision distance differences are immaterial and do not affect selection;
- all persisted NPZ artifacts load safely with `allow_pickle=False`;
- development history was deliberately separated from the compact submission dependency closure.
