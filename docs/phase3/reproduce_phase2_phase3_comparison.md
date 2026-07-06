# Reproduce Phase 2 vs Phase 3 comparison

This guide defines the exact files needed to compare the Phase 2 quantum reservoir, the Phase 3 RF-QRC probe, and the ESN reference on the same volatility task.

## Goal

This workflow currently regenerates these row-level prediction files:

- `results/reproduction/phase2_phase3_comparison/phase2_feedback_tfim_qrc_reference_predictions.csv`
- `results/qrc/rf_qrc/phase3_rf_qrc_tail_probe_predictions_level_rate.csv`

These regenerated files support comparison of:

- actual `future_rv_20d`
- Phase 2 best QRC
- Phase 3 RF-QRC second-encode ring
- train-calibrated q80, q90, q95 thresholds

The historical ESN export remains available under `archive/phase2/results/`,
but regeneration is pending explicit output-directory support.

## Phase 2 ESN export script

The historical ESN prediction export script is retained at:

```text
archive/phase2/scripts/export_esn_predictions.py
```

The reproduction workflow should write regenerated comparison outputs under:

```text
results/reproduction/phase2_phase3_comparison/
```

The archived script currently retains its historical `results/tables/` default
and therefore must not be run unchanged as part of this guide until its output
directory is made explicit.

## Run order

Run from the repository root.

```bash
python scripts/data/prepare_phase2_spy_vix_dataset.py

mkdir -p results/reproduction/phase2_phase3_comparison

python archive/phase2/scripts/run_phase2_feedback_tfim_qrc_reference.py \
  --output-dir results/reproduction/phase2_phase3_comparison

python scripts/qrc/rf_qrc/run_phase3_rf_qrc_tail_probe.py \
  --input-mode level_rate \
  --leak 0.3 \
  --ridge-alpha 3000 \
  --results-dir results/reproduction/phase2_phase3_comparison

# Phase 2 ESN export remains pending explicit output-directory support.
```

## Expected output files

Check that the files exist:

```bash
ls -lh results/reproduction/phase2_phase3_comparison/phase2_feedback_tfim_qrc_reference_predictions.csv
ls -lh results/qrc/rf_qrc/phase3_rf_qrc_tail_probe_predictions_level_rate.csv
```

Expected columns:

`phase2_feedback_tfim_qrc_reference_predictions.csv`

```text
split
date
actual_future_rv_20d
qrc_pred_future_rv_20d
run_name
```

`phase3_rf_qrc_tail_probe_predictions_level_rate.csv`

```text
split
date
actual_future_rv_20d
rf_qrc_single_encode_level_rate_pred
rf_qrc_second_encode_level_rate_pred
rf_qrc_second_encode_ring_level_rate_pred
```

Historical ESN export schema:

`archive/phase2/results/phase2_esn_predictions.csv`

```text
date
actual_future_rv_20d
esn_pred_future_rv_20d
run_name
```

## Interpretation

The RF-QRC internal ablation is:

```text
single encode -> second encode all-pairs -> second encode ring
```

The project-level comparison is:

```text
Phase 2 best QRC -> Phase 3 RF-QRC ring -> ESN
```

Do not use the RF-QRC single-encode control as a substitute for the Phase 2 best QRC. They are different reservoirs.

## Reproducibility checks

Use train thresholds from the current volatility dataset:

```python
q80 = train["actual_future_rv_20d"].quantile(0.80)
q90 = train["actual_future_rv_20d"].quantile(0.90)
q95 = train["actual_future_rv_20d"].quantile(0.95)
```

Typical thresholds from the Phase 2/Phase 3 dataset are approximately:

```text
q80 = 0.213
q90 = 0.275
q95 = 0.326
```

If regenerated results differ materially from the Phase 2 paper, check in this order:

1. dataset file: `data/processed/phase2_spy_vix_volatility.csv`
2. chronological split implementation
3. PCA component count and train-only PCA fitting
4. random seeds
5. log-target readout setting
6. ridge alpha / reservoir configuration

## Figure to produce

The primary comparison figure should overlay test-period time series:

```text
actual future_rv_20d
Phase 2 best QRC: qrc_pred_future_rv_20d
Phase 3 RF-QRC ring: rf_qrc_second_encode_ring_level_rate_pred
ESN: esn_pred_future_rv_20d
train q80/q90/q95 horizontal thresholds
```

The secondary figure should compare:

```text
RMSE
correlation
prediction standard deviation
q80 F1
q90 F1
q95 F1
top-20 predicted/actual tail ratio
```
