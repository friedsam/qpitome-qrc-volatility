# Result organization

Results are separated by evidentiary role.

- `canonical/` — standardized comparison outputs intended to be the project source of truth.
- `reference/` — retained Phase 2 and other historical benchmark tables used for comparison.
- `tuning/` — parameter-search trials and selected configurations.
- `experimental/` — exploratory forecasting probes and notebook-generated analyses.
- `diagnostics/` — mechanism, scaling, geometry, memory, and ablation studies.
- `hardware/` — finite-shot, transfer, readiness, and actual QPU outputs.
- `archive/` — superseded material retained for provenance.

`results/tables/` is legacy and should not receive new files. Existing material is moved gradually and conservatively so historical references are not broken unnecessarily.

Generated bulky run artifacts remain local according to `.gitignore` unless deliberately promoted into one of the structured result areas.
