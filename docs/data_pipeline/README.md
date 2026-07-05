# Data pipeline status - living document

Last updated: 2026-07-05

## Current phase

**RESET / DATA FOUNDATION**

Historical work is archived on `archive/pre-reset-20260705`. Active work is on `reset/data-first-20260705`.

## Dataset registry

| Dataset | Role | Public/reproducible | Status |
|---|---|---:|---|
| `paper_monthly` | Primary challenge-grounded benchmark | Yes | collection + preparation scripts ready; raw retrieval pending |
| `volare` | Rich realized-volatility comparison | Access-dependent | immutable import script ready |
| `legacy_daily` | Historical SPY/VIX comparison only | Yes | immutable import script ready; old CSV not tracked in archive |

## Completed

- [x] Pre-reset state preserved on archive branch.
- [x] New clean reset branch created from a deliberately minimal tree.
- [x] Challenge and supplied paper stack reread.
- [x] Internet/primary-source grounding round completed.
- [x] Data quality protocol defined.
- [x] Feature research summary defined.
- [x] Dataset-separated data and result contracts defined.
- [x] General dataset audit script implemented and smoke-tested.
- [x] Paper raw-data collector implemented.
- [x] Paper target/core preparation script implemented and synthetic Yahoo-format smoke-tested.
- [x] Close and adjusted-close target definitions preserved for explicit comparison.
- [x] Both quarterly/annual RV conventions preserved until paper parity is resolved.
- [x] VOLARE and legacy immutable import scripts smoke-tested.
- [x] Cutoff-aware exploration script implemented and smoke-tested.

## Current blockers / unresolved definitions

These remain explicit and must not be guessed:

- actual paper-monthly raw retrieval must run in an internet-enabled project environment;
- exact Shiller source-field mapping for DP and EP;
- exact default-spread definition used by the anchor paper/Bucci benchmark;
- publication-availability lag treatment for CPI and industrial production;
- exact quarterly/annual log-RV convention used in the authors' implementation;
- VOLARE export must be supplied from the user's authorized access;
- legacy daily CSV was ignored by Git and therefore is not recoverable from the archive branch.

## In progress

- [ ] Retrieve the `paper_monthly` raw snapshot.
- [ ] Run preparation, audit, and exploration on the actual primary data.
- [ ] Import and audit a VOLARE export.
- [ ] Import and audit the frozen legacy daily CSV if it still exists locally/qBraid.
- [ ] Resolve the four paper-parity definitions above from primary/reference sources.

## Modeling gate

No substantial model comparison begins until the primary dataset has:

- provenance manifest;
- schema report;
- missingness/quality report;
- prepared table;
- exploration summary.

After that, modeling proceeds in parallel streams:

1. cheap classical sanity baseline;
2. ESN - primary classical reservoir baseline;
3. TFIM control;
4. Rydberg simulator -> finite shots -> Aquila.

HAR may be retained as a diagnostic reference but does not define the target task.

## Change log

### 2026-07-05

Reset initiated after identifying task drift, inconsistent ESN selection objectives, diagnostic censoring mishandled as NaN, and excessive repository clutter. The new pipeline is dataset-first and dataset-separated.

The reset branch was reduced to a minimal root, then only reviewed data-pipeline documents and tested scripts were added back. No historical result directories or parallel experimental scripts were carried into active work.
