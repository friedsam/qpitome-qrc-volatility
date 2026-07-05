# Data pipeline status - living document

Last updated: 2026-07-05

## Current phase

**RESET / DATA FOUNDATION**

Historical work is archived on `archive/pre-reset-20260705`. Active work is on `reset/data-first-20260705`.

## Dataset registry

| Dataset | Role | Public/reproducible | Status |
|---|---|---:|---|
| `paper_monthly` | Primary challenge-grounded benchmark | Yes | collection scaffold created |
| `volare` | Rich realized-volatility comparison | Access-dependent | import scaffold created |
| `legacy_daily` | Historical SPY/VIX comparison only | Yes | import scaffold created |

## Completed

- [x] Pre-reset state preserved on archive branch.
- [x] New clean reset branch created.
- [x] Challenge and supplied paper stack reread.
- [x] Internet/primary-source grounding round completed.
- [x] Data quality protocol defined.
- [x] Feature research summary defined.
- [x] Dataset-separated folder contract defined.

## In progress

- [ ] Run paper-monthly raw data collection locally.
- [ ] Import available VOLARE export without altering source files.
- [ ] Import legacy daily dataset as comparison only.
- [ ] Run dataset audits and review every missingness category.
- [ ] Produce exploration summaries per dataset.

## Next modeling gate

No substantial model comparison begins until the primary dataset has:

- provenance manifest;
- schema report;
- missingness/quality report;
- prepared table;
- exploration summary.

After that, modeling proceeds in parallel streams:

1. cheap classical sanity baseline;
2. ESN;
3. TFIM control;
4. Rydberg simulator -> finite shots -> Aquila.

## Change log

### 2026-07-05

Reset initiated after identifying task drift, inconsistent ESN selection objectives, diagnostic censoring mishandled as NaN, and excessive repository clutter. The new pipeline is dataset-first and dataset-separated.
