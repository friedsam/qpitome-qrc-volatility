# Transition-Forecasting Data Pipeline Workflow

## Scope

This document is the living workflow record for the final Phase 3 transition-forecasting pipeline. It must be updated whenever a pipeline stage is added, removed, frozen, or materially changed.

The current implementation target is the canonical processed dataset only. Model-specific three-channel inputs are deferred until the canonical one-channel dataset has been rebuilt and compared against the historical pre-cleaning lineage.

## Governing principles

- Raw source files are immutable.
- Automated retrieval is attempted first.
- Repository fallback data may be used only after live retrieval or validation failure and must be labeled in the acquisition manifest.
- Structural OHLC corrections occur before volatility, event, matching, split, or tensor construction.
- No interpolation, forward fill, backward fill, winsorization, arbitrary clipping, synthetic dates, or silent repair is allowed.
- Reusable logic belongs in `src/`; scripts are thin command-line wrappers.
- Canonical processed data live under `data/processed/`, not `results/`.
- Run outputs, logs, diagnostics, and run manifests live under the matching `results/<topic>/<script_name>/<run_id>/` hierarchy.
- A candidate dataset is not promoted until validation and before/after lineage checks pass.
- The final control ratio is currently frozen at three controls per positive because that is the implemented and historically evaluated contract.

## Stage 0: raw-source acquisition

### Purpose

Acquire the public global stock-index OHLC dataset reproducibly while preserving a verified fallback snapshot for temporary source or network failures.

### Inputs

- Kaggle dataset: `guillemservera/global-stock-indices-historical-data`
- Repository fallback snapshot:
  `data/fallback/transition_forecasting/global_stock_indices_historical_data/`
- Fallback integrity manifest:
  `data/fallback/transition_forecasting/global_stock_indices_historical_data/fallback_manifest.json`

### Current implementation files

- `scripts/transition_forecasting/quality/fetch_global_index_ohlc.py`

### Target implementation files

- `src/transition_forecasting/data/acquisition.py`
- `scripts/transition_forecasting/data/fetch_global_index_ohlc.py`
- `tests/transition_forecasting/data/test_acquisition.py`

### Processing

1. Attempt live Kaggle download in `auto` mode.
2. Validate combined and individual-index schemas.
3. On live retrieval or validation failure, verify fallback hashes.
4. Copy the verified fallback snapshot into the raw-data destination.
5. Record requested mode, actual mode, live error, source URL, license, timestamps, validation results, and file hashes.

### Created files

- `data/raw/transition_forecasting/global_stock_indices_historical_data/all_indices_data.csv`
- `data/raw/transition_forecasting/global_stock_indices_historical_data/individual_indices_data/*.csv`
- `data/raw/transition_forecasting/global_stock_indices_historical_data/source_manifest.json`
- `data/raw/transition_forecasting/global_stock_indices_historical_data/raw_acquisition_manifest.json`

## Stage 1: raw OHLC parsing and structural cleaning

### Purpose

Create a deterministic cleaned OHLC lineage while preserving every removed-row reason and leaving the raw files unchanged.

### Inputs

- Raw individual-index CSV files from Stage 0.
- Frozen structural bad-print policy.

### Current implementation files

- `src/transition_forecasting/quality/structural_bad_prints.py`
- `scripts/transition_forecasting/quality/build_canonical_clean_ohlc.py`

### Target implementation files

- `src/transition_forecasting/data/cleaning.py`
- `src/transition_forecasting/quality/structural_bad_prints.py`
- `scripts/transition_forecasting/data/build_cleaned_ohlc.py`
- `tests/transition_forecasting/data/test_cleaning.py`
- `tests/transition_forecasting/quality/test_structural_bad_prints.py`

### Processing

1. Normalize column names.
2. Parse dates and OHLC values.
3. Drop invalid dates.
4. Drop missing, nonnumeric, zero, or negative OHLC values.
5. Drop rows with `high < low`.
6. Sort by date and keep the last duplicate-date row.
7. Apply the causal structural bad-print detector using only preceding observations.
8. Drop flagged structural bad-print rows.
9. Record every removed row in `row_corrections.csv`.

### Frozen structural checks for the current source lineage

- 12 structural rows flagged.
- 2 affected indices.

These counts are lineage guards, not universal constants. A raw-source hash change must trigger a source-lineage drift failure rather than forcing old counts onto new data.

### Candidate outputs

- cleaned individual-index OHLC files in managed temporary storage;
- `row_corrections.csv`;
- source/output file manifest;
- cleaning manifest with policy and counts.

## Stage 2: OHLC inventory and effective-start assessment

### Purpose

Determine which index histories are eligible and establish the effective start used for volatility and event construction.

### Current implementation files

- `src/transition_forecasting/quality/global_index_ohlc_audit.py`
- `src/transition_forecasting/quality/global_range_quality.py`
- `scripts/transition_forecasting/quality/audit_global_index_ohlc.py`
- `scripts/transition_forecasting/quality/audit_global_ohlc_range_quality.py`

### Inputs

- Cleaned individual-index OHLC files from Stage 1.

### Outputs

- global OHLC inventory;
- eligibility decisions;
- recommended effective start per index;
- range-quality summary.

## Stage 3: canonical daily volatility

### Purpose

Build the daily log Parkinson-volatility series used by all downstream transition stages.

### Inputs

- Cleaned OHLC from Stage 1.
- Effective starts from Stage 2.

### Current implementation

- Volatility construction is embedded in `scripts/transition_forecasting/build_processed_dataset.py`.
- `log_parkinson` is defined in `src/transition_forecasting/catalogue/transition_events.py`.

### Target implementation files

- `src/transition_forecasting/data/volatility.py`
- `tests/transition_forecasting/data/test_volatility.py`

### Output

- candidate `daily_volatility.csv.gz` with:
  - index;
  - date;
  - `log_parkinson_volatility`;
  - effective start;
  - source lineage metadata where practical.

## Stage 4: transition catalogue

### Purpose

Detect persistent volatility-transition onsets and cluster them into global episodes.

### Inputs

- Cleaned OHLC and effective starts.

### Implementation files

- `src/transition_forecasting/catalogue/transition_events.py`
- `src/transition_forecasting/catalogue/global_transition_catalogue.py`
- `scripts/transition_forecasting/catalogue/build_global_transition_catalogue.py`
- `tests/transition_forecasting/catalogue/test_transition_events.py`

### Frozen event logic

- pre-2016 threshold-estimation history;
- 80th percentile threshold per market;
- persistent high-volatility condition;
- causal prior-window requirement;
- 60-observation separation;
- 7-calendar-day global clustering;
- representative market selection within global episodes.

### Outputs

- raw transition catalogue;
- clustered transition catalogue;
- representative transition catalogue;
- global episode catalogue;
- transition-count summary.

## Stage 5: sample and matched-control construction

### Purpose

Create the transition-forecasting dataset with causal origins, future volatility paths, and matched calm controls.

### Inputs

- Representative transition catalogue.
- Cleaned OHLC inventory and effective starts.

### Implementation files

- `src/transition_forecasting/modeling/global_stage_d_dataset.py`
- `src/transition_forecasting/catalogue/transition_events.py`
- `scripts/transition_forecasting/modeling/build_global_stage_d_dataset.py`
- `tests/transition_forecasting/modeling/test_global_stage_d_dataset.py`

### Frozen sample contract

- 40-observation input window;
- 10-observation future target path;
- leads 1, 5, and 10;
- controls matched within index, lead, and split;
- three controls per positive;
- no control-origin reuse within an index/lead/split stratum;
- 60-observation exclusion from detected onsets;
- matching features: level, 5/20 means, 5/20 slopes, 20-observation standard deviation, and 20-observation maximum;
- one global episode may appear in only one split;
- historical development split preserved initially:
  - train before 2013;
  - validation 2013–2015;
  - test 2016 onward;
- test remains unevaluated during development.

### Outputs

- `sample_manifest.csv`;
- one-channel sequence tensor `(n_samples, 40, 1)`;
- matching-balance diagnostics;
- dataset summary.

### Balance-reporting requirement

Report both pooled and within-stratum balance. Do not quote only pooled maximum absolute SMD. Current historical evidence indicates excellent pooled balance but materially weaker balance in some small strata.

## Stage 6: canonical dataset assembly and candidate validation

### Purpose

Assemble the single canonical processed dataset without overwriting the last valid dataset until all checks pass.

### Current implementation

- `scripts/transition_forecasting/build_processed_dataset.py`
- `scripts/transition_forecasting/validate_processed_dataset.py`
- `scripts/transition_forecasting/analyze_cleanup_evidence.py`

### Target implementation files

- `src/transition_forecasting/data/dataset.py`
- `src/transition_forecasting/data/validation.py`
- `scripts/transition_forecasting/data/build_processed_dataset.py`
- `scripts/transition_forecasting/data/validate_processed_dataset.py`
- `tests/transition_forecasting/data/test_processed_dataset.py`

### Canonical output path

`data/processed/transition_forecasting/global_transition_dataset/`

### Required canonical files

- `cleaned_ohlc.csv.gz`
- `daily_volatility.csv.gz`
- `transition_catalogue.csv`
- `sample_manifest.csv`
- `sequence_tensors.npz`
- `row_corrections.csv`
- `manifest.json`

### Required validation before promotion

- required files exist;
- raw source hashes recorded;
- raw files unchanged;
- OHLC validity and uniqueness;
- daily-volatility finiteness and uniqueness;
- unique sample IDs;
- tensor/manifest ID identity;
- tensor shape `(n, 40, 1)` and finite values;
- exactly three controls per positive;
- no control-origin reuse within matching strata;
- no episode split leakage;
- no test evaluation;
- structural correction counts consistent with the cleaning manifest;
- no forbidden transformations;
- deterministic rebuild under identical inputs and software;
- before/after event and sample delta relative to the historical pre-cleaning lineage;
- every downstream change attributable to the structural corrections and deterministic rebuilding.

### Promotion rule

Build in managed temporary storage. Promote atomically only after validation. Remove temporary artifacts automatically. Do not retain `.previous`, `_new`, `_fixed`, `_v2`, or other duplicate canonical trees.

## Deferred Stage 7: model-input transformation

This stage is intentionally deferred until the corrected canonical one-channel dataset has been validated against the historical lineage.

The expected deterministic model representation is:

1. normalized volatility level;
2. first difference;
3. deterministic time coordinate.

The derived three-channel data may be stored under `data/processed/` once frozen. It must remain reconstructable from the canonical one-channel tensor and recorded preprocessing parameters.

## Data sources

### Primary source

- Kaggle dataset: `guillemservera/global-stock-indices-historical-data`
- Public dataset page: `https://www.kaggle.com/datasets/guillemservera/global-stock-indices-historical-data`
- Dataset author states that underlying daily OHLCV data were sourced from Yahoo Finance.
- Repository-declared license: CC-BY-NC-4.0.

### Fallback source

- Byte-preserved repository snapshot of the last successful public raw acquisition.
- Used only after live retrieval or validation failure.
- Verified by `fallback_manifest.json` hashes.
- Must never contain manually cleaned or altered processed data.

## Run and result-path convention

For each executable script:

```text
src/<topic>/*.py
scripts/<topic>/<script_name>.py
results/<topic>/<script_name>/<run_id>/*
tests/<topic>/test_*.py
```

Canonical datasets remain under `data/processed/`; only run logs, manifests, diagnostics, and comparison evidence belong under `results/`.
