# GPT-2 Dataset Pipeline Map

Purpose: reconstruct the corrected production dataset from the authoritative GPT-2 procedure without importing unrelated repository history or downstream experimental variants.

## Authoritative source chain

### 1. Raw source contract

Input directory:

`data/raw/transition_forecasting/global_stock_indices_historical_data/individual_indices_data/`

Relevant source files:

- `src/transition_forecasting/quality/global_index_ohlc_audit.py`
- `src/transition_forecasting/quality/ohlc_range_quality.py`
- `src/transition_forecasting/quality/global_range_quality.py`

Historical artifacts consumed by GPT-2:

- `results/transition_forecasting/quality/global_index_ohlc_audit/global_index_audit_001/global_index_ohlc_inventory.csv`
- `results/transition_forecasting/quality/global_ohlc_range_quality/global_range_quality_001/global_range_quality.csv`

Contract:

- Audit every raw CSV.
- Require valid date, high, and low fields.
- Determine eligibility from usable OHLC fraction, history length, duplicate dates, and invalid ranges.
- Exclude the consolidated panel and VIX from transition construction.
- Derive each market's effective start from the first calendar year with at least 95% nonzero intraday ranges.

The historical inventory and effective-start table are part of the GPT-2 procedure. Recomputing them from a transformed intermediate is not automatically equivalent.

### 2. OHLC loading and volatility transform

Authoritative functions:

- `src/transition_forecasting/catalogue/global_transition_catalogue.py::_load_generic_ohlc`
- `src/transition_forecasting/catalogue/transition_events.py::log_parkinson`

Procedure:

1. Read the selected index file.
2. Resolve date, high, and low columns case-insensitively.
3. Parse dates with `pandas.to_datetime(..., errors="coerce")`.
4. Parse high and low numerically.
5. Drop missing values.
6. Retain positive ranges with `High >= Low`.
7. Keep the final row for duplicate dates.
8. Sort chronologically.
9. Apply the market-specific effective start.
10. Compute
   `abs(log(High / Low)) / sqrt(4 log 2)`.
11. Replace zero volatility with missing and take the natural logarithm.

Correction insertion point:

- Detect and remove the 12 frozen structural bad-print rows after parsing and duplicate resolution but before the volatility transform.
- Do not alter any other valid row.
- Do not interpolate, fill, winsorize, clip, or synthesize dates.

### 3. Per-market transition detection

Authoritative files:

- `src/transition_forecasting/catalogue/transition_events.py`
- `src/transition_forecasting/catalogue/global_transition_catalogue.py`

Frozen constants:

- Training threshold period: dates before 2016-01-01.
- Minimum pre-2016 history: 250 volatility rows.
- Threshold: 80th percentile of pre-2016 log Parkinson volatility.
- Persistence requirement: at least 10 high observations in the next 15 sessions.
- Prior calm requirement: at most 2 high observations in the preceding 10 sessions.
- Minimum separation within a market: 60 volatility rows.
- Stage D history floor: onset position at least 40.

Detection order:

- Iterate candidate positions chronologically.
- Require the candidate position itself to be high.
- Apply persistence, prior-calm, and 60-session separation checks.
- Once accepted, update the market-local last-onset position.

### 4. Global episode clustering

Authoritative function:

`src/transition_forecasting/catalogue/global_transition_catalogue.py::_cluster_dates`

Procedure:

- Sort unique onset dates globally.
- Start a new episode when a date is more than seven calendar days after the first date of the current episode.
- This is a fixed window anchored to the cluster start, not chain clustering from the most recent date.
- Assign sequential IDs `GE001`, `GE002`, ... in chronological order.

### 5. Representative-market selection

Authoritative function and tables:

- `build_global_transition_catalogue`
- `MARKET_GROUPS`
- `REPRESENTATIVE_PRIORITY`

Procedure:

1. Sort raw events by onset date, market group, and index.
2. Attach the global episode ID.
3. Rank duplicate indices within a market group using the frozen priority lists.
4. Sort by episode ID, market group, representative rank, and onset date.
5. Keep the first event for each `(episode_id, market_group)`.

Priority groups:

- United States: GSPC, NYA, DJI, IXIC, RUT, XAX.
- Australia: AORD, AXJO.
- China mainland: Shanghai, Shenzhen.
- All single-index market groups keep their sole index.

The representative catalogue is the direct input to Stage D.

### 6. Positive sample construction

Authoritative file:

`src/transition_forecasting/modeling/global_stage_d_dataset.py`

Frozen constants:

- Leads: 1, 5, 10 sessions.
- Input window: 40 sessions ending at origin.
- Target horizon: next 10 sessions.
- Features: level, mean5, mean20, slope5, slope20, std20, max20.

Procedure:

- Rebuild each index's volatility series using the effective start stored in the representative catalogue.
- Locate each representative onset exactly in the volatility index.
- For each lead, set `origin = onset - lead`.
- Require causal features, a complete 40-session input, and a complete 10-session target.
- Generate one positive row per eligible `(representative event, lead)`.
- Positive ordering follows representative-catalogue row order, then lead order `(1, 5, 10)`.

### 7. Episode-level chronological split

Authoritative functions:

- `_split`
- `_episode_split_assignments`

Cutoffs:

- train: episode earliest onset before 2013-01-01.
- validation: episode earliest onset from 2013-01-01 through 2015-12-31.
- test: episode earliest onset on or after 2016-01-01.

Every event and matched control belonging to an episode inherits that one episode-level split. No episode may span splits.

### 8. Control candidate construction and matching

Authoritative Stage D procedure:

- Group positives by `(index, lead, split)`.
- Candidate origins run from position 39 through the last position with a complete 10-session target.
- Candidate origin's own date must lie in the group's split.
- Reject candidates within 60 volatility rows of any representative onset for that index.
- Require all seven causal matching features.
- Standardize candidates using the candidate pool's own feature mean and standard deviation; replace zero scales with one.
- Compute Euclidean distance in the seven standardized features.
- Process positives in their existing group order.
- Select nearest unused candidate positions.
- Select exactly three controls per positive.
- Candidate reuse is prohibited within each `(index, lead, split)` group, but the original canonical Stage D procedure does not globally prohibit reuse across different leads.

Later fold-local chronology/rematching modules are downstream modeling derivatives and are not part of canonical Stage D reconstruction.

### 9. Manifest and tensor ordering

Procedure:

- Concatenate all positive rows first, followed by all control rows.
- Control rows are produced in grouped positive-processing order and nearest-candidate pick order.
- Build one `(40, 1)` tensor for every manifest row in manifest order.
- Persist exact `input_start_date`, `origin_date`, and `target_end_date`.
- Manifest and tensor sample IDs must align exactly.

### 10. Required validation

The canonical dataset must prove:

- no episode split overlap;
- exactly three controls for every positive;
- manifest/tensor one-to-one alignment;
- shape `(n_samples, 40, 1)`;
- complete 10-session targets;
- no test evaluation during development;
- correction ledger identifies exactly 12 removed rows across exactly two indices;
- no transformations beyond the documented row removals and GPT-2 loading rules.

## Source-of-truth boundary

Canonical construction ends with:

- representative global catalogue;
- Stage D sample manifest;
- Stage D sequence tensors;
- Stage D balance diagnostics.

The following are not canonical dataset construction:

- chronological fold-local control rematching;
- repaired rolling datasets;
- cross-market tensor experiments;
- ESN representation screens;
- Rydberg assays;
- parity-control compatibility code.

## Current reconstruction finding

The current catalogue and Stage D source implementations are materially the same as the authoritative GPT-2 implementations. The observed no-correction count drift therefore originates upstream of transition detection, most likely from reconstructing the historical inventory/effective-start contract from rewritten cleaned files rather than applying the correction inside the exact GPT-2 raw-to-volatility lineage.

The production fix should preserve this map and change the narrowest upstream layer necessary. It should not patch event counts, episode IDs, samples, or controls after construction.
