# QRC Volatility Forecasting - Data-First Reset

Active branch: `reset/data-first-20260705`

Historical Phase 1-3 work is preserved on `archive/pre-reset-20260705`.

This branch is intentionally minimal. The project is rebuilt from the data upward, with strict dataset separation and no parallel-script proliferation.

## Scientific objective

Test whether an analog Rydberg quantum reservoir can extract forecasting value that survives finite sampling and hardware noise, relative primarily to a classical echo-state network on a challenge-grounded volatility task.

## Model stream

1. Persistence / AR / HAR / HARX / Ridge - classical diagnostics.
2. GARCH(1,1)-t - integrated econometric volatility baseline.
3. ESN - primary classical reservoir baseline.
4. LSTM - required nonlinear sequence baseline; integration pending.
5. TFIM - preliminary quantum control/reference.
6. Rydberg QRC - primary quantum architecture; simulator, finite shots, then Aquila early.

The frozen monthly development task and current model status are documented under `docs/data_pipeline/` and `docs/model_references/`.

## Dataset separation

- `paper_monthly`: primary benchmark derived from the finance-track anchor paper.
- `volare`: comparison dataset using richer realized-volatility estimators; access-dependent.
- `legacy_daily`: old SPY/VIX dataset retained only for comparison and regression checks.

No results are written outside `results/<dataset>/...`.

See `docs/data_pipeline/README.md` for the living status record.
