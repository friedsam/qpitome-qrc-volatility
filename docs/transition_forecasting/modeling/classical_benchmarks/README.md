# Classical Transition Benchmarks

This benchmark family evaluates only the agreed classical comparators on the active rematched transition dataset. It does not run QRC models.

## Dataset contract

Point `--dataset-root` to the extracted `global_transition_dataset_1d` directory containing:

```text
purged_walk_forward_folds/rematched_rolling_manifest.csv
purged_walk_forward_folds/rematched_rolling_tensors.npz
cleaned_ohlc.csv.gz
```

Folds 4–6 are used for model/readout selection. Folds 7–8 are frozen confirmation folds. Test rows are not evaluated.

Reported groups are `Transition`, `L1`, `L5`, `L10`, `Controls`, and `Pooled`. Controls are never subdivided by lead.

## Linear baselines

```bash
python scripts/transition_forecasting/modeling/classical_benchmarks/run_linear.py \
  --dataset-root /path/to/global_transition_dataset_1d \
  --run-id linear_current_data_001
```

This produces persistence, canonical HAR, and direct sequence-ridge forecasts.

## GARCH

```bash
python scripts/transition_forecasting/modeling/classical_benchmarks/run_garch.py \
  --dataset-root /path/to/global_transition_dataset_1d \
  --run-id garch_current_data_001
```

The runner preserves the existing `arch` Student-t GARCH(1,1) backend. If `arch` is unavailable, it uses the deterministic SciPy implementation of the same zero-mean Student-t GARCH(1,1) specification and records the backend in diagnostics.

## Direct ESN

The ESN run is intentionally resumable. Run selection once, then each fold in a fresh process, then finalize.

```bash
python scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py select \
  --dataset-root /path/to/global_transition_dataset_1d \
  --run-id esn_current_data_001
```

Use the printed run directory for the remaining commands:

```bash
for fold in 4 5 6 7 8; do
  python scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py predict-fold \
    --dataset-root /path/to/global_transition_dataset_1d \
    --run-dir results/transition_forecasting/modeling/classical_benchmarks/esn/esn_current_data_001 \
    --fold "$fold"
done

python scripts/transition_forecasting/modeling/classical_benchmarks/run_esn.py finalize \
  --dataset-root /path/to/global_transition_dataset_1d \
  --run-dir results/transition_forecasting/modeling/classical_benchmarks/esn/esn_current_data_001
```

The primary ESN is a direct future-path model. `esn_shuffled` is its temporal-order control. No HAR residual is used.

## Transition-first ESN tuning

The bounded tuning runner preserves the established `level_diff_time` representation, `final_mean_std` pooling, and washout 10. It searches only the previously established short-memory and persistent-memory ESN regimes, uses folds 4–6 for selection, and applies sequence-ridge pooled/control/RMSE guardrails plus an ordered-versus-shuffled transition guardrail.

```bash
python scripts/transition_forecasting/modeling/classical_benchmarks/run_esn_tuning.py \
  --dataset-root /path/to/global_transition_dataset_1d \
  --run-id esn_transition_tuned_001
```

The verified run selected a five-seed direct ESN with 300 units, spectral radius 0.55, input scale 0.20, leak 0.95, connectivity 0.02, and ridge alpha 120000. The result family is:

```text
results/transition_forecasting/modeling/classical_benchmarks/esn_tuning/<run_id>/
```

The tuning result is kept separate from the original three-candidate ESN benchmark; it does not overwrite prior predictions or selection records. The selected specification is based only on folds 4–6, but folds 7–8 had already been viewed during earlier bounded tuning iterations. The stored folds 7–8 result must therefore be described as a confirmation recheck, not as a pristine one-shot holdout.

## Canonical comparison

```bash
python scripts/transition_forecasting/modeling/classical_benchmarks/run_canonical.py \
  --linear-run results/transition_forecasting/modeling/classical_benchmarks/linear/linear_current_data_001 \
  --garch-run results/transition_forecasting/modeling/classical_benchmarks/garch/garch_current_data_001 \
  --esn-run results/transition_forecasting/modeling/classical_benchmarks/esn/esn_current_data_001 \
  --run-id classical_current_data_001
```

The canonical comparison restricts every model to the exact intersection of finite `(fold, sample_id)` predictions before calculating the submission tables.

## Main artifacts

Every individual run contains row-level predictions, configuration and dataset hashes, a submission metrics table, fold and horizon diagnostics, runtime, and a summary. The canonical run additionally contains exact-common-row predictions, coverage, and deltas against sequence ridge.
