# Data pipeline status - living document

Last updated: 2026-07-05

## Current phase

**RESET / DATA FOUNDATION**

Historical work is archived on `archive/pre-reset-20260705`. Active work is on `reset/data-first-20260705`.

## Dataset registry

| Dataset | Role | Public/reproducible | Status |
|---|---|---:|---|
| `paper_monthly` | Primary challenge-grounded benchmark | Yes | real raw snapshot collected; preparation and audit completed |
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

From the repository root and active environment:

```bash
python scripts/data/prepare_paper_dataset.py \
  --snapshot data/raw/paper_monthly/20260705T170000Z
```

The preparation script owns the derived outputs. Do not create benchmark slices with ad hoc CLI commands.

Then audit the full prepared table:

```bash
python scripts/data/audit_dataset.py \
  --input data/interim/paper_monthly/prepared_core.parquet \
  --outdir results/paper_monthly/data_audit \
  --dataset-name paper_monthly
```

Primary outputs:

```text
data/interim/paper_monthly/prepared_core.parquet
data/interim/paper_monthly/daily_source_prepared.parquet
data/interim/paper_monthly/missingness_categories.csv
data/interim/paper_monthly/preparation_metadata.json
data/processed/paper_monthly/paper_parity_1950_02_to_2017_12.parquet
data/processed/paper_monthly/extended_complete_market_months.parquet
results/paper_monthly/data_audit/manifest.json
results/paper_monthly/data_audit/schema.csv
results/paper_monthly/data_audit/quality_report.csv
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
- [x] Close and adjusted-close target equivalence verified on real data.
- [x] Both quarterly/annual RV conventions preserved until paper parity is resolved.
- [x] VOLARE and legacy immutable import scripts smoke-tested.
- [x] Cutoff-aware exploration script implemented and smoke-tested.

## Current blockers / unresolved definitions

These remain explicit and must not be guessed:

- exact Shiller source-field mapping for DP and EP;
- exact default-spread definition used by the anchor paper/Bucci benchmark;
- publication-availability lag treatment for CPI and industrial production;
- exact quarterly/annual log-RV convention used in the authors' implementation;
- VOLARE export must be supplied from the user's authorized access;
- legacy daily CSV was ignored by Git and therefore is not recoverable from the archive branch.

## In progress

- [ ] Make the preparation script emit the paper-parity and extended processed outputs directly.
- [ ] Run cutoff-aware EDA on the real primary data.
- [ ] Resolve the remaining paper-parity definitions from primary/reference sources.
- [ ] Import and audit a VOLARE export when access returns.
- [ ] Import and audit the frozen legacy daily CSV if it still exists locally.

## Modeling gate

No substantial model comparison begins until the primary dataset has:

- provenance record;
- schema report;
- missingness/quality report;
- prepared table;
- exploration summary.

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
