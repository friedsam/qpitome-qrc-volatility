# QRC Volatility Forecasting - Data-First Reset

Active branch: `reset/data-first-20260705`

Historical Phase 1-3 work is preserved on `archive/pre-reset-20260705`.

This branch is intentionally minimal. The project is rebuilt from the data upward, with strict dataset separation and no parallel-script proliferation.

## Scientific objective

Test whether an analog Rydberg quantum reservoir can extract forecasting value that survives finite sampling and hardware noise, relative primarily to a classical echo-state network on a challenge-grounded volatility task.

## Model stream

1. Cheap classical sanity baseline - diagnostic only, not the target.
2. ESN - primary classical reservoir baseline.
3. TFIM - preliminary quantum control/reference.
4. Rydberg QRC - primary quantum architecture; simulator, finite shots, then Aquila early.

HAR may be retained as a diagnostic reference where informative, but it does not define the target task.

## Dataset separation

- `paper_monthly`: primary benchmark derived from the finance-track anchor paper.
- `volare`: comparison dataset using richer realized-volatility estimators; access-dependent.
- `legacy_daily`: old SPY/VIX dataset retained only for comparison and regression checks.

No results are written outside `results/<dataset>/...`.

See `docs/data_pipeline/README.md` for the living status record.
