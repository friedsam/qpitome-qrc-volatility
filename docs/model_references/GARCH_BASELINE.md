# GARCH baseline — integrated development reference

Status: **integrated for the frozen monthly development task**.

## Role

GARCH is the standard econometric conditional-variance baseline required by the challenge. It is evaluated on the same monthly target and 245 development forecast dates as the other models.

## Frozen development specification

- model: GARCH(1,1);
- mean: zero;
- innovations: Student-t;
- input: daily S&P 500 log returns from the same `close` series used to construct the monthly target;
- information boundary: daily observations through the prediction origin, inclusive;
- history: protocol `train_calendar_start` through the prediction origin, including the origin month used to condition the forecast;
- refit: every forecast fold;
- forecast: analytic multi-step daily conditional-variance path for the next calendar month;
- monthly variance: sum of daily conditional-variance forecasts;
- target mapping: `0.5 * log(monthly conditional variance)`;
- trading-day count: deterministic NYSE calendar, not future observed rows.

## Development result

On the 245-date development protocol:

| Metric | Value |
|---|---:|
| RMSE log RV | 0.347405 |
| MAE log RV | 0.276056 |
| QLIKE variance | 0.258247 |
| MZ intercept | -0.322170 |
| MZ slope | 0.917250 |
| MZ R² | 0.536298 |
| Converged folds | 245 / 245 |

Interpretation:

- weaker than HARX/ESN on log-RV RMSE;
- strongest current baseline on QLIKE;
- therefore retained as a genuinely complementary comparator rather than a redundant point-forecast baseline.

A separate expanding-history probe produced slightly better RMSE but slightly worse QLIKE. The current window is retained because its QLIKE is better and the RMSE difference is negligible. This choice is frozen for the development comparison unless a specific later result justifies reopening it.

## Run

Install the project environment, then run:

```bash
bash scripts/run_garch_baseline.sh
```

Equivalent direct command:

```bash
python scripts/models/run_garch.py
```

Outputs:

```text
results/challenge_primary/models/garch/
├── predictions.csv
├── metrics.csv
└── run_manifest.json
```

## Remaining work

- rerun on the final 2018+ temporal holdout after model choices are frozen;
- include GARCH in final diagnostic plots and benchmark tables;
- report its QLIKE advantage explicitly alongside RMSE rather than collapsing model comparison to one metric.
