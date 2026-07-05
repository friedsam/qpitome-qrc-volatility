# Data quality protocol

Every dataset must pass this protocol before feature engineering or modeling.

## 1. Provenance and immutability

- Save raw downloads unchanged under `data/raw/<dataset>/`.
- Record source URL, retrieval timestamp, requested date range, provider symbol/series ID, file hash, and license/access notes.
- Never overwrite raw files. New retrievals receive new dated snapshots.

## 2. Schema and type audit

For every table:

- row count and column count;
- exact column names and units;
- inferred and enforced dtypes;
- date/time timezone and frequency;
- duplicate rows and duplicate timestamps;
- monotonicity of time index;
- impossible values and domain constraints.

## 3. Missingness audit

For every column:

- count and fraction missing;
- first and last missing date;
- contiguous missing runs;
- whether missingness is raw, alignment-induced, rolling-window warm-up, future-target censoring, or intentionally right-censored.

Rules:

- Raw missing values are never silently filled.
- Rolling warm-up NaNs are expected but explicitly tagged.
- Future-target NaNs are retained until the final modeling frame is constructed and then excluded by rule.
- Right-censored diagnostics are stored with a value plus a censoring flag, never encoded as NaN alone.
- Any model-specific row loss must be reported and date-aligned against every comparator.

## 4. Numerical integrity

Check:

- positive/negative infinity;
- invalid logs and ratios;
- zero or negative prices where impossible;
- extreme one-day returns and provider adjustments;
- split/dividend adjustment consistency;
- range consistency (`low <= open/close <= high` where applicable).

## 5. Time alignment and leakage

- Define information-availability time for every feature.
- Lag exogenous variables according to publication availability, not only observation month, where feasible.
- Use no future information in scaling, PCA, feature selection, model selection, or imputation.
- Purge label-overlap where targets span future periods.
- Train/validation/test dates must be identical across compared models.

## 6. Distribution and dependence checks

Before modeling:

- summary statistics and robust quantiles;
- target histogram and log-target comparison;
- outlier/event table;
- autocorrelation and partial autocorrelation on non-overlapping and overlapping constructions separately;
- regime/era plots without assigning causal meaning;
- feature-target relationships using train-only or predeclared exploratory periods.

## 7. Output artifacts

Each prepared dataset must produce:

- `manifest.json` - provenance, hashes, retrieval metadata;
- `schema.csv` - units, dtypes, source/derived status;
- `quality_report.csv` - missingness, infinities, duplicates, ranges;
- `prepared.parquet` - model-ready table;
- `exploration_summary.md` - concise findings and caveats.

No model run is allowed unless these artifacts exist.
