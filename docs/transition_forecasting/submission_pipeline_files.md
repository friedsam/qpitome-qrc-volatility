# Submission Pipeline File Inventory

## Purpose

This is the living allow-list for files used by the final Phase 3 transition-forecasting pipeline. Update it whenever a required file is added, removed, renamed, transferred, or replaced.

A file belongs here only when it is required to acquire data, build the canonical processed dataset, validate it, derive frozen model inputs, run final baselines/QRC, reproduce reported results, or document the reproducible submission.

## Status labels

- `keep`: already suitable or expected to remain.
- `refactor`: logic is useful but must be moved behind a thin wrapper or otherwise cleaned.
- `review`: potentially useful; inclusion depends on dependency or scientific validation.
- `defer`: not required for the current canonical data build.
- `exclude`: historical, exploratory, duplicated, agent-only, or otherwise not part of the final pipeline.

## Current working branch

- `stage1-dev-ext`

No new branch is planned. The final clean submission set will later be transferred selectively rather than merging the full branch.

## Data-source files

| Status | Path | Role |
|---|---|---|
| keep | `data/fallback/transition_forecasting/global_stock_indices_historical_data/fallback_manifest.json` | Verifies the fallback raw snapshot. |
| keep | `data/fallback/transition_forecasting/global_stock_indices_historical_data/source_manifest.json` | Records public source identity and licensing. |
| keep | `data/fallback/transition_forecasting/global_stock_indices_historical_data/all_indices_data.csv` | Unmodified combined raw panel retained as fallback. |
| keep | `data/fallback/transition_forecasting/global_stock_indices_historical_data/individual_indices_data/*.csv` | Unmodified individual-market raw OHLC fallback files. |
| generated | `data/raw/transition_forecasting/global_stock_indices_historical_data/*` | Live or fallback raw acquisition output; not hand-curated processed data. |
| generated | `data/processed/transition_forecasting/global_transition_dataset/*` | Canonical processed dataset. |

## Configuration

| Status | Path | Role |
|---|---|---|
| review | `config/transition_forecasting/global_expansion.yaml` | Contains global-data parameters; retain only fields used by the final pipeline. |
| review | `config/transition_forecasting/stage_a.yaml` | Earlier stage configuration; likely superseded by a single final data config. |
| target | `config/transition_forecasting/data_pipeline.yaml` | Preferred single frozen configuration if configuration extraction is needed during refactor. Do not create until parameters are mapped. |

## Source modules: acquisition, cleaning, and canonical data

| Status | Path | Role |
|---|---|---|
| target | `src/transition_forecasting/data/__init__.py` | Data-pipeline package. |
| target | `src/transition_forecasting/data/acquisition.py` | Live-first acquisition, fallback verification, source validation, and acquisition manifest. |
| target | `src/transition_forecasting/data/cleaning.py` | Parsing, basic row validation, duplicate handling, correction ledger, and cleaned OHLC construction. |
| target | `src/transition_forecasting/data/volatility.py` | Canonical log Parkinson-volatility construction. |
| target | `src/transition_forecasting/data/dataset.py` | End-to-end candidate assembly and atomic promotion. |
| target | `src/transition_forecasting/data/validation.py` | Semantic dataset validation and historical delta checks. |
| keep | `src/transition_forecasting/quality/structural_bad_prints.py` | Frozen causal structural bad-print policy; may receive narrow API cleanup. |
| review | `src/transition_forecasting/quality/global_index_ohlc_audit.py` | OHLC inventory and eligibility. |
| review | `src/transition_forecasting/quality/global_range_quality.py` | Effective-start/range-quality logic; inspect before transfer. |
| review | `src/transition_forecasting/quality/early_ohlc_quality.py` | Supporting audit; retain only if used by the final effective-start workflow. |

## Source modules: transition catalogue and samples

| Status | Path | Role |
|---|---|---|
| keep | `src/transition_forecasting/catalogue/__init__.py` | Catalogue package. |
| keep | `src/transition_forecasting/catalogue/transition_events.py` | Core volatility/event/matching constants and functions; split unrelated plotting/output helpers if needed. |
| keep | `src/transition_forecasting/catalogue/global_transition_catalogue.py` | Global transition detection, clustering, and representative-event selection. |
| review | `src/transition_forecasting/catalogue/global_transition_validation.py` | Catalogue validation; include only checks used by the canonical workflow. |
| keep | `src/transition_forecasting/modeling/global_stage_d_dataset.py` | Positive/control construction, split assignment, tensors, and matching diagnostics; rename only if necessary and without changing semantics. |
| review | `src/transition_forecasting/modeling/stage_d_candidate_pool.py` | Include only if required by the final rebuilt sample path. |
| defer | `src/transition_forecasting/modeling/chronological_splits.py` | Rolling-fold confirmation, not required for initial historical split reproduction. |
| defer | `src/transition_forecasting/modeling/chronological_control_matching.py` | Rolling-fold confirmation/rematching. |
| defer | `src/transition_forecasting/modeling/chronological_rematched_dataset.py` | Rolling-fold confirmation dataset. |

## Scripts: target thin wrappers

| Status | Path | Role |
|---|---|---|
| refactor | `scripts/transition_forecasting/quality/fetch_global_index_ohlc.py` | Current acquisition implementation; move logic to `src` and replace with thin wrapper under matching data path. |
| refactor | `scripts/transition_forecasting/quality/build_canonical_clean_ohlc.py` | Current cleaning implementation; move logic to `src`. |
| refactor | `scripts/transition_forecasting/build_processed_dataset.py` | Current orchestration-heavy builder; replace with a thin wrapper around one `src` entry point. |
| refactor | `scripts/transition_forecasting/validate_processed_dataset.py` | Move validation logic to `src`. |
| review | `scripts/transition_forecasting/analyze_cleanup_evidence.py` | Keep only if the final validation workflow consumes its evidence package. |
| keep/thin | `scripts/transition_forecasting/catalogue/build_global_transition_catalogue.py` | Catalogue wrapper; verify it contains no scientific implementation. |
| keep/thin | `scripts/transition_forecasting/modeling/build_global_stage_d_dataset.py` | Dataset wrapper; verify it contains no scientific implementation. |
| target | `scripts/transition_forecasting/data/fetch_global_index_ohlc.py` | Final acquisition wrapper. |
| target | `scripts/transition_forecasting/data/build_cleaned_ohlc.py` | Final cleaning wrapper if a separately executable stage remains useful. |
| target | `scripts/transition_forecasting/data/build_processed_dataset.py` | Final canonical build wrapper. |
| target | `scripts/transition_forecasting/data/validate_processed_dataset.py` | Final semantic validation wrapper. |
| review | `scripts/runs/run_submission.py` | Final top-level orchestrator; refactor only after individual stages work. |

## Tests required for the canonical data pipeline

| Status | Path | Role |
|---|---|---|
| target | `tests/transition_forecasting/data/test_acquisition.py` | Live/fallback behavior, fallback corruption, manifest mode labeling, total failure. |
| target | `tests/transition_forecasting/data/test_cleaning.py` | Invalid rows, duplicates, immutable raw data, correction ledger, deterministic output. |
| target | `tests/transition_forecasting/data/test_volatility.py` | Parkinson formula, effective starts, finite/ordered output. |
| target | `tests/transition_forecasting/data/test_processed_dataset.py` | End-to-end fallback build, canonical files, ratios, IDs, split integrity, hashes, deterministic rebuild. |
| keep/update | `tests/transition_forecasting/quality/test_cluster_and_early_ohlc_quality.py` | Fixed-anchor cluster test and early-data checks. |
| keep/update | `tests/transition_forecasting/quality/test_global_index_ohlc_audit.py` | Inventory/eligibility checks. |
| keep/update | `tests/transition_forecasting/catalogue/test_transition_events.py` | Event-onset and catalogue behavior. |
| keep/update | `tests/transition_forecasting/modeling/test_global_stage_d_dataset.py` | Matching, 3:1 ratio, no reuse, episode split integrity, tensor construction. |
| exclude | `tests/data/test_transition_pipeline.py` | Historical eight-index/results-based integration test; not the canonical final pipeline. |

## Documentation required for the data pipeline

| Status | Path | Role |
|---|---|---|
| keep/update | `docs/transition_forecasting/pipeline_workflow.md` | Living stage-by-stage workflow, sources, inputs, outputs, and files. |
| keep/update | `docs/transition_forecasting/submission_pipeline_files.md` | This living exact file inventory. |
| keep/update | `docs/transition_forecasting_data_contract.md` | Canonical dataset contract; correct the erroneous 5:1 control ratio and reconcile with the final workflow. |
| keep/update | `docs/structural_ohlc_quality_policy.md` | Frozen structural correction policy and evidence. |
| review | `docs/data_pipeline_completion_plan.md` | Historical planning document; retire or mark superseded once the new workflow is implemented. |
| keep/update | `docs/transition_forecasting/stage_a_task_freeze.md` | Retain only if it documents still-binding event/data choices. |

## Canonical processed-data outputs

Generated under:

`data/processed/transition_forecasting/global_transition_dataset/`

Required:

- `cleaned_ohlc.csv.gz`
- `daily_volatility.csv.gz`
- `transition_catalogue.csv`
- `sample_manifest.csv`
- `sequence_tensors.npz`
- `row_corrections.csv`
- `manifest.json`

The initial rebuilt tensor remains `(n_samples, 40, 1)` for direct historical comparison.

## Run outputs for the canonical data build

Expected path symmetry:

`results/transition_forecasting/data/build_processed_dataset/<run_id>/`

Expected run artifacts:

- `run_manifest.json`
- command/stage logs;
- `processed_dataset_audit.json`;
- historical before/after sample delta;
- historical before/after event delta;
- correction and source-lineage summary;
- deterministic hash comparison;
- warnings or unresolved provenance notes.

These are run artifacts, not canonical datasets.

## Deferred model-input files

Do not implement until the corrected canonical one-channel dataset is validated.

Likely target files:

- `src/transition_forecasting/modeling/sequence_inputs.py`
- `scripts/transition_forecasting/modeling/build_sequence_inputs.py`
- `tests/transition_forecasting/modeling/test_sequence_inputs.py`
- `data/processed/transition_forecasting/sequence_inputs/` or another single frozen processed-data location.

Expected representation:

- normalized volatility level;
- first difference;
- deterministic time coordinate.

## Deferred model and QRC files

These remain outside the current data-build scope but are candidates for later selective integration:

- `src/baselines/numpy_esn.py`
- `src/baselines/esn_representation.py`
- final linear/HAR baseline module;
- final field-test ESN module;
- constrained ESN module;
- final Rydberg reservoir module;
- shared evaluation metrics module;
- thin run wrappers and tests under matching paths.

Exploratory Stage E screens, duplicated nested script directories, old branching experiments, and unused historical result families are not automatically included.

## Current frozen decisions

- Work on `stage1-dev-ext`; do not create another branch unless a concrete safety issue requires it.
- Do not merge this branch wholesale into `main` or `stage1-dev`.
- Use three controls per positive.
- Preserve the historical train/validation/test split for initial reconstruction.
- Keep canonical tensors one-channel during cleaning validation.
- Delay three-channel model inputs until historical comparison passes.
- Retain the verified fallback raw snapshot and document its use.
- Minimize acquisition testing to the critical failure and provenance paths.
- Do not promote a new canonical dataset until before/after lineage evidence is complete.
