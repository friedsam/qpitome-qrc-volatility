# Data pipeline status - living document

Last updated: 2026-07-05

## Current phase

**RESET / DATA FOUNDATION**

Historical work is archived on `archive/pre-reset-20260705`. Active work is on `reset/data-first-20260705`.

## Dataset registry

| Dataset | Role | Public/reproducible | Status |
|---|---|---:|---|
| `paper_monthly` | Primary challenge-grounded benchmark | Yes | real raw snapshot collected; preparation, audit, EDA, and preliminary feature engineering implemented |
| `volare` | Rich realized-volatility comparison | Access-dependent | webpage currently unavailable; deferred, not blocking primary work |
| `legacy_daily` | Historical SPY/VIX comparison only | Yes | immutable import script ready; old CSV not tracked in archive |

## Proven real raw snapshot

The first real primary snapshot was assembled on 2026-07-05 under:

```text
data/raw/paper_monthly/20260705T170000Z/
```

Expected raw files:

```text
gspc_daily_yahoo.json
fred_TB3MS_three_month_tbill.csv
fred_CPIAUCSL_cpi.csv
fred_INDPRO_industrial_production.csv
fred_AAA_aaa_corporate_yield.csv
fred_BAA_baa_corporate_yield.csv
ff3_monthly.zip
short_term_reversal_monthly.zip
```

Raw files are immutable inputs. Do not edit their contents.

### Public source URLs

Yahoo S&P 500 daily history is collected by `scripts/data/collect_paper_dataset.py` from the Yahoo chart endpoint for `^GSPC`, beginning 1950-01-01.

FRED direct CSV downloads:

```text
https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS
https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL
https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDPRO
https://fred.stlouisfed.org/graph/fredgraph.csv?id=AAA
https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA
```

Kenneth French direct ZIP downloads:

```text
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_CSV.zip
```

Save the files with the exact expected names above. The automated collector exists, but the FRED endpoint proved unreliable from the current Mac environment; manual download of the public files is the proven fallback and does not change the pipeline after raw acquisition.

## Reproduce the primary data pipeline

### 1. Prepare

```bash
python scripts/data/prepare_paper_dataset.py \
  --snapshot data/raw/paper_monthly/20260705T170000Z
```

The preparation script owns all derived outputs, including benchmark slices. Do not create benchmark slices with ad hoc CLI commands.

### 2. Audit

```bash
python scripts/data/audit_dataset.py \
  --input data/interim/paper_monthly/prepared_core.parquet \
  --outdir results/paper_monthly/data_audit \
  --dataset-name paper_monthly
```

### 3. Explore the exact paper-parity table

```bash
python scripts/data/explore_dataset.py \
  --input data/processed/paper_monthly/paper_parity_1950_02_to_2017_12.parquet \
  --dataset-name paper_monthly_paper_parity \
  --target log_rv_close \
  --outdir results/paper_monthly/eda/paper_parity
```

The EDA removes exact numeric duplicates, target duplicates, and future-target columns from feature rankings. It reports target-state persistence separately from exogenous feature families.

### 4. Build preliminary feature families

```bash
python scripts/data/engineer_paper_features.py
```

This creates deterministic, leakage-aware predictors only. It does **not** fit a scaler, PCA, imputer, or supervised feature selector.

Preliminary families:

- `volatility_state`: current log RV plus 3- and 12-month memory;
- `volatility_dynamics`: 1- and 3-month changes, acceleration, short-minus-long state, prior-window robust surprise;
- `market_drivers`: market excess, SMB, HML, short-term reversal;
- `credit_rates`: T-bill, AAA, BAA, and BAA-minus-AAA state plus 1-month changes;
- `macro_dynamics`: lagged inflation and industrial-production growth plus their 1-month changes.

Frequency-domain features remain deferred until a null-tested construction is defined.

## Primary outputs

```text
data/interim/paper_monthly/prepared_core.parquet
data/interim/paper_monthly/daily_source_prepared.parquet
data/interim/paper_monthly/missingness_categories.csv
data/interim/paper_monthly/preparation_metadata.json
data/processed/paper_monthly/paper_parity_1950_02_to_2017_12.parquet
data/processed/paper_monthly/extended_complete_market_months.parquet
data/processed/paper_monthly/features/preliminary_features.parquet
data/processed/paper_monthly/features/feature_catalog.csv
data/processed/paper_monthly/features/feature_manifest.json
results/paper_monthly/data_audit/manifest.json
results/paper_monthly/data_audit/schema.csv
results/paper_monthly/data_audit/quality_report.csv
results/paper_monthly/eda/paper_parity/exploration_summary.md
```

## Real-data findings from the first run

Full prepared table:

- 919 monthly rows;
- 38 columns;
- 1950-01-31 through 2026-07-31;
- no duplicate rows;
- no duplicate dates;
- monotonic monthly dates;
- no bad date parses;
- no infinite values.

The 2026-07 row contains only two daily returns and is an incomplete current month. It must not be used as an ordinary monthly observation.

The exact anchor-paper calendar slice is:

```text
1950-02-28 through 2017-12-31
```

This produces exactly 815 monthly observations, matching the anchor paper's stated sample size.

Within that 815-row slice, the only missing values are expected initialization effects:

- 3-month rolling RV features: 1 leading row missing;
- 12-month rolling RV features: 10 leading rows missing;
- one leading row missing in each lagged macro-growth availability variant.

Close and adjusted-close produce identical monthly RV values for all 919 overlapping months in this `^GSPC` source. The duplicate definitions remain in the audit table for provenance, but they should not be treated as independent features.

Observed source missingness near the current right edge:

- 2025-10: CPI missing in the source;
- 2026-06: CPI, industrial production, and French factors not yet present in the downloaded source files;
- 2026-07: current incomplete market month and broad right-edge source missingness.

No source gap is silently filled or dropped.

First EDA findings on the 815-month paper-parity table:

- monthly log RV has strong decaying memory: lag-1 ACF about 0.695, lag-12 about 0.359;
- short-, medium-, and long-memory volatility states all relate to next-month volatility;
- the transparent BAA-minus-AAA spread candidate is the strongest current exogenous relationship among the available public inputs;
- CPI and industrial-production level relationships are treated as suspect because secular trends can create spurious rank association;
- extreme observations include both isolated shocks and prolonged high-volatility clusters, motivating separate state and dynamics channels.

## Completed

- [x] Pre-reset state preserved on archive branch.
- [x] New clean reset branch created from a deliberately minimal tree.
- [x] Challenge and supplied paper stack reread.
- [x] Internet/primary-source grounding round completed.
- [x] Data quality protocol defined.
- [x] Feature research summary defined.
- [x] Dataset-separated data and result contracts defined.
- [x] General dataset audit script implemented and smoke-tested.
- [x] Real paper raw snapshot collected.
- [x] Real paper preparation run completed.
- [x] Real full-table audit completed.
- [x] Exact 815-month anchor-paper calendar slice verified.
- [x] Preparation script now emits paper-parity and extended processed outputs directly.
- [x] Close and adjusted-close target equivalence verified on real data.
- [x] Redundancy-aware real-data EDA implemented.
- [x] Preliminary deterministic feature engineering implemented.
- [x] Both quarterly/annual RV conventions preserved until paper parity is resolved.
- [x] VOLARE and legacy immutable import scripts smoke-tested.

## Current blockers / unresolved definitions

These remain explicit and must not be guessed:

- exact Shiller source-field mapping for DP and EP;
- exact default-spread definition used by the anchor paper/Bucci benchmark;
- publication-availability lag treatment for CPI and industrial production;
- exact quarterly/annual log-RV convention used in the authors' implementation;
- VOLARE export must be supplied from the user's authorized access;
- legacy daily CSV was ignored by Git and therefore is not recoverable from the archive branch.

## In progress

- [ ] Run and inspect the preliminary feature-engineering output on real data.
- [ ] Audit redundancy/correlation structure of engineered features without promoting features from full-sample correlations.
- [ ] Resolve the remaining paper-parity definitions from primary/reference sources.
- [ ] Define the common train/validation/test protocol.
- [ ] Import and audit a VOLARE export when access returns.
- [ ] Import and audit the frozen legacy daily CSV if it still exists locally.

## Modeling gate

No substantial model comparison begins until the primary dataset has:

- provenance record;
- schema report;
- missingness/quality report;
- prepared table;
- exploration summary;
- reproducible preliminary feature table and catalog.

After that, modeling proceeds in parallel streams:

1. cheap classical sanity baseline;
2. ESN - primary classical reservoir baseline;
3. Phase 2-equivalent TFIM control;
4. Rydberg simulator -> finite shots -> Aquila.

HAR may be retained as a diagnostic reference but does not define the target task.

## Change log

### 2026-07-05

Reset initiated after identifying task drift, inconsistent ESN selection objectives, diagnostic censoring mishandled as NaN, and excessive repository clutter. The new pipeline is dataset-first and dataset-separated.

The reset branch was reduced to a minimal root, then only reviewed data-pipeline documents and tested scripts were added back. No historical result directories or parallel experimental scripts were carried into active work.

The first real `paper_monthly` snapshot was collected manually after public-source download failures in the automated collector. The full dataset was prepared and audited. The exact 815-month paper calendar slice was verified. Documentation was updated so another researcher can reconstruct the raw snapshot and rerun preparation and audit without relying on chat history.

The existing preparation script was updated to emit both the exact 815-month paper-parity table and an extended completed-market-month table directly. The earlier ad hoc CLI slice is no longer part of the reproducible workflow.

The EDA script was corrected to remove exact duplicates and features identical to the target from exploratory rankings. Preliminary feature engineering was then added as one documented script organized around volatility state, dynamics, market drivers, credit/rates, and lagged macro dynamics.
