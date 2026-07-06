# Reproduce Phase 2 vs Phase 3 comparison

This guide defines the exact files needed to compare the Phase 2 quantum reservoir, the Phase 3 RF-QRC probe, and the ESN reference on the same volatility task.

## Goal

Generate these row-level prediction files:

- `results/tables/phase2_feedback_tfim_qrc_reference_predictions.csv`
- `results/tables/phase3_rf_qrc_tail_probe_predictions_level_rate.csv`
- `results/tables/phase2_esn_predictions.csv`

These files support the paper-style comparison:

- actual `future_rv_20d`
- Phase 2 best QRC
- Phase 3 RF-QRC second-encode ring
- ESN reference
- train-calibrated q80, q90, q95 thresholds

## One-time restore of ESN export script

The ESN prediction export script was archived/removed during Phase 2 cleanup. Restore it from the pre-cleanup commit:

```bash
git checkout 99b5bab -- scripts/export_esn_predictions.py
```

This restores `scripts/export_esn_predictions.py`, which writes:

- `results/tables/phase2_esn_predictions.csv`
- `results/tables/phase2_esn_prediction_export_metrics.csv`

## Run order

Run from the repository root.

```bash
python scripts/data/prepare_phase2_spy_vix_dataset.py
python archive/phase2/scripts/run_phase2_feedback_tfim_qrc_reference.py
python scripts/qrc/rf_qrc/run_phase3_rf_qrc_tail_probe.py --input-mode level_rate --leak 0.3 --ridge-alpha 3000
python scripts/export_esn_predictions.py
```

## Expected output files

Check that the files exist:

```bash
ls -lh results/tables/phase2_feedback_tfim_qrc_reference_predictions.csv
ls -lh results/tables/phase3_rf_qrc_tail_probe_predictions_level_rate.csv
ls -lh results/tables/phase2_esn_predictions.csv
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

`phase2_esn_predictions.csv`

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
