# Classical transition benchmarks

This is the frozen classical comparison for the current one-channel rematched transition dataset. It contains no QRC model.

## Models

- persistence;
- canonical HAR, ridge alpha 100;
- direct 40-session sequence ridge, alpha 3000;
- zero-mean Student-t GARCH(1,1) using the `arch` backend;
- tuned direct ESN and its identically specified shuffled-order control.

The frozen model authority is:

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
```

The qBraid submission workflow never retunes these models and never evaluates the reserved test partition.

## Reporting

Every model reports `Transition`, `L1`, `L5`, `L10`, `Controls`, and `Pooled` rows with RMSE, log-volatility QLIKE, and Mincer-Zarnowitz intercept, slope, and R². Controls are not subdivided by lead. The canonical table uses the exact intersection of finite `(fold, sample_id)` predictions across all models.

## Standalone result mapping

```text
results/transition_forecasting/modeling/classical_benchmarks/<model>/run/<run-id>/
```

## qBraid aggregate mapping

```text
results/runs/<run-id>/files/transition_forecasting/modeling/classical_baselines/<model>/run/<run-id>/
```

## Canonical qBraid workflow

```bash
.venv/bin/python scripts/runs/run_submission.py financial-classical \
  --run-id <literal-run-id> \
  --transition-source-mode <mode-from-preflight>
```

The aggregate workflow builds and validates the data first, then runs the three classical producer families, creates the exact-common-row canonical comparison, and writes `classical_baseline_audit.json`. It stops on the first failed command.

The ESN tuning history remains development provenance. It is not called by the submission workflow; the submitted ESN specification is frozen in configuration.
