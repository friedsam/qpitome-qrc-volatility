# Transition-Forecasting Processed Dataset Contract

## Purpose

This document defines the non-negotiable construction and validation rules for the canonical transition-forecasting dataset. It is the control surface for human and automated agents. A workflow may change implementation details, but it may not silently change these rules.

## Canonical output

The only canonical processed dataset is:

`data/processed/transition_forecasting/global_transition_dataset/`

Required files:

- `cleaned_ohlc.csv.gz`
- `daily_volatility.csv.gz`
- `transition_catalogue.csv`
- `sample_manifest.csv`
- `sequence_tensors.npz`
- `row_corrections.csv`
- `manifest.json`

`results/` contains run provenance and historical artifacts, not canonical data.

## Proven parity point

The pre-correction global Stage D pipeline was reproduced from the original individual-index OHLC data and compared with:

`results/transition_forecasting/modeling/stage_d_dataset/20260718T010839Z`

Verified parity:

- 4,572 rows in reference and reconstruction.
- Identical sample-ID set.
- No mismatches in any original manifest field.
- Identical tensor keys.
- Identical tensor shape `(4572, 40, 1)`.
- Maximum tensor absolute difference `0.0`.
- Bit-for-bit tensor equality.

The reconstructed manifest adds `input_start_date` and `target_end_date` as provenance fields. These do not alter samples or tensors.

## Data-cleaning rules

Allowed automatic removals:

- invalid dates;
- missing or nonnumeric OHLC values;
- zero or negative OHLC values;
- rows where `high < low`;
- duplicate dates, keeping the last source row;
- rows flagged by the frozen causal structural-bad-print policy.

Forbidden transformations:

- interpolation;
- forward fill;
- backward fill;
- synthetic dates;
- winsorization;
- arbitrary clipping;
- replacement of invalid values;
- silent row repair.

Every removed row must be recorded in `row_corrections.csv` with source identity, date or source row, reason, and action.

## Ordering constraints

Structural bad-print removal must occur before:

1. effective-start assessment;
2. Parkinson-volatility calculation;
3. event detection;
4. episode clustering;
5. representative-event selection;
6. positive-window construction;
7. control matching;
8. split and tensor output.

Controls must be rematched after any upstream change. Old controls may not be retained against changed series or events.

## Matching contract

- Sequence window: 40 observations.
- Forecast horizon: 10 observations.
- Five controls per positive.
- Controls matched within index, lead, and split.
- A control origin may not be reused within an index/lead/split stratum.
- Event-exclusion rules remain causal.
- Matching is deterministic for identical inputs.
- Every control must reference an existing positive sample.
- Every positive must have exactly five controls.

## Split contract

- Allowed splits: `train`, `val`, `test`.
- A global episode may appear in exactly one split.
- No episode split leakage is permitted.

## Tensor contract

- `sequence_tensors.npz` must load with `allow_pickle=False` for canonical outputs.
- It must contain at least `X` and `sample_id`.
- `X` must have shape `(n_samples, 40, 1)`.
- All tensor values must be finite.
- Tensor sample IDs and manifest sample IDs must be identical as sets.
- Sample IDs must be unique.

## Publication contract

- Raw data are immutable.
- Construction occurs in temporary storage.
- A failed build must not leave a partial canonical directory.
- The canonical directory is replaced only after build-time validation.
- The standard workflow then runs `validate_processed_dataset.py` and records `processed_dataset_audit.json` in the run directory.
- A failed audit makes the workflow fail and the dataset must not be treated as submission-ready.

## Required automated checks

The post-build audit verifies:

- all required files exist;
- OHLC schema, positivity, ordering, uniqueness, and range consistency;
- daily-volatility finiteness and uniqueness;
- manifest schema and unique IDs;
- valid labels and splits;
- no episode leakage;
- exact 5:1 control ratio and linkage;
- tensor shape, finiteness, and ID agreement;
- correction counts agree with the manifest;
- forbidden transformations remain disabled;
- file hashes and row counts are recorded.

Matching tests additionally verify deterministic construction, exact controls per positive, no control reuse, incomplete-control rejection, and split-leakage rejection.

## Change control

Time constraints may require multiple related fixes in one commit. This does not relax validation. Any combined change must still provide:

- explicit documentation of all behavioral changes;
- test coverage for the changed behavior;
- a machine-readable audit report;
- exact before/after counts where applicable;
- no silent methodological changes.

A successful command exit alone is not evidence of dataset validity. The dataset is submission-ready only when build validation, automated tests, and the post-build audit pass.
