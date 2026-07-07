# Phase 3 RF-QRC findings

This note records the current Phase 3 result after restoring the Phase 2 final QRC baseline, rebuilding the original forecast-aware regime layer, comparing it to the Phase 3 RF-QRC ring probe and ESN reference, and running the first bounded RF-QRC ring leak/ridge sweep.

## Current status

The Phase 3 RF-QRC ring is not a uniformly better volatility forecaster. It is a more selective crisis detector.

The empirical pattern is stable across the raw forecast comparison, the forecast-aware regime layer, and the leak/ridge sweep:

- Phase 2 final QRC remains the better calibrated quantum volatility baseline.
- Phase 3 RF-QRC ring substantially improves q95 crisis detection.
- ESN remains the strongest classical reference overall.
- RF-QRC ring appears to create a structural calibration/tail tradeoff rather than a simple hyperparameter tuning issue.

## Main comparison: raw forecast metrics

| Model | RMSE | repo-style QLIKE | Corr | Actual std | Pred std | q80 F1 | q90 F1 | q95 F1 | Top-20 pred/actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 2 final QRC | 0.095082 | -2.229724 | 0.434624 | 0.104783 | 0.054796 | 0.562500 | 0.132450 | 0.000000 | 0.430313 |
| Phase 3 RF-QRC ring | 0.105438 | -1.040723 | 0.666724 | 0.104783 | 0.067121 | 0.397436 | 0.358025 | 0.413793 | 0.542389 |
| ESN reference | 0.071896 | -2.504775 | 0.728883 | 0.104783 | 0.080431 | 0.672199 | 0.479638 | 0.575000 | 0.705398 |

Interpretation:

- RF-QRC ring improves correlation and tail overlap relative to the Phase 2 final QRC.
- RF-QRC ring degrades RMSE and repo-style QLIKE, so it is not a better calibrated continuous forecaster.
- Its raw q95 amplitude F1 improvement is large: Phase 2 QRC has zero q95 amplitude F1, while RF-QRC ring reaches 0.414.

## Main comparison: forecast-aware regime layer

The old Phase 2 q95 result was not raw q95 amplitude F1. It came from the original forecast-aware regime layer: forecast thresholds combined with contemporaneous stress-confirmation flags. Reapplying that same layer gives the fair operational comparison.

### q95 crisis-like detection

| Model | Precision | Recall | F1 | Signal rate | Event rate |
|---|---:|---:|---:|---:|---:|
| Phase 2 final QRC | 0.155 | 0.265 | 0.196 | 0.0569 | 0.0334 |
| Phase 3 RF-QRC ring | 0.463 | 0.559 | 0.507 | 0.0402 | 0.0334 |
| ESN reference | 0.228 | 0.853 | 0.360 | 0.1246 | 0.0334 |

RF-QRC ring gives the cleanest q95 crisis-like signal among the three models. It more than doubles Phase 2 QRC crisis F1 and has substantially higher precision than ESN, while firing less often than Phase 2 QRC.

### q90 warning-or-higher detection

| Model | Precision | Recall | F1 | Signal rate | Event rate |
|---|---:|---:|---:|---:|---:|
| Phase 2 final QRC | 0.289 | 0.640 | 0.398 | 0.241 | 0.109 |
| Phase 3 RF-QRC ring | 0.618 | 0.378 | 0.469 | 0.0667 | 0.109 |
| ESN reference | 0.377 | 0.802 | 0.513 | 0.232 | 0.109 |

RF-QRC ring remains highly selective. It improves precision but loses broad q90 recall.

### q80 watch-or-higher detection

| Model | Precision | Recall | F1 | Signal rate | Event rate |
|---|---:|---:|---:|---:|---:|
| Phase 2 final QRC | 0.448 | 0.828 | 0.581 | 0.443 | 0.239 |
| Phase 3 RF-QRC ring | 0.483 | 0.770 | 0.594 | 0.382 | 0.239 |
| ESN reference | 0.481 | 0.791 | 0.598 | 0.394 | 0.239 |

At the broad watch level, RF-QRC ring is competitive with both baselines.

## Diagnosis

The RF-QRC ring shifts the inductive bias away from continuous volatility calibration and toward sparse crisis detection.

The time-series behavior shows the mechanism: the RF-QRC ring undershoots calm and moderate-volatility periods, then reacts strongly around crisis windows. It behaves more like a crisis-warning head than a continuous volatility forecaster.

This is not just a poor choice of ridge alpha or leaky rate. A bounded leak/ridge sweep showed a strong tradeoff: settings that improve calibration suppress q95 crisis detection, while settings that preserve q95 crisis F1 keep poor QLIKE. The tradeoff appears structural for the current ring feature map.

## Hyperparameter sweep note

The first sweep notebook initially reported QLIKE with the wrong convention. The standalone RF-QRC script used level-style QLIKE:

```python
mean(log(yhat) + y / yhat)
```

The Phase 2 comparison uses repo-style variance QLIKE:

```python
mean(log(yhat**2) + y**2 / yhat**2)
```

Saved sweep predictions remain usable, but any sweep table column named `qlike` should be treated as wrong-scale unless recomputed as repo-style variance QLIKE.

Repo-style QLIKE function:

```python
def repo_qlike(y_true, y_pred, eps=1e-12):
    target = np.maximum(np.asarray(y_true, dtype=float) ** 2, eps)
    pred = np.maximum(np.asarray(y_pred, dtype=float) ** 2, eps)
    return float(np.mean(np.log(pred) + target / pred))
```

The corrected conclusion remains qualitative: QLIKE and q95 F1 are anti-correlated across the tested grid.

## Scientific interpretation

The RF-QRC ring has value, but not as a single scalar volatility forecaster. Its useful role is crisis discrimination.

Working interpretation:

> The RF-QRC ring changes the reservoir's inductive bias from calibrated volatility tracking toward sparse crisis detection. In the original forecast-aware regime layer, this produces a large q95 improvement, but the raw forecast is poorly calibrated. This suggests that indiscriminate entanglement creates a structural tail/calibration tradeoff.

## Next experiments

### 1. Hybrid regime model

Fastest practical next step:

- Use Phase 2 final QRC for calibrated baseline / broad watch-warning behavior.
- Use RF-QRC ring as a crisis trigger.
- Compare hybrid rules against Phase 2 QRC, RF-QRC ring, and ESN.

Candidate rules:

1. Phase 2 regime only.
2. RF-QRC regime only.
3. Hybrid A: Phase 2 watch/warning plus RF-QRC crisis-like.
4. Hybrid B: max severity of Phase 2 and RF-QRC regimes.
5. Hybrid C: crisis-like if RF-QRC crisis-like or both models are warning-or-higher.
6. ESN reference.

Success criterion: improve q95 crisis detection relative to Phase 2 while preserving better q80/q90 behavior than RF-QRC alone.

### 2. Structured level-rate RF-QRC

More scientific next step:

The current RF-QRC input uses level/rate features, but the ring entangler does not preserve level/rate structure. It treats the concatenated vector as generic adjacent qubits:

```text
[level_1, level_2, level_3, rate_1, rate_2, rate_3]
```

Structured alternative:

```text
q0, q1, q2 = level qubits
q3, q4, q5 = rate qubits
```

Test structured entanglers:

```text
cross_matched:
(q0, q3), (q1, q4), (q2, q5)

cross_all:
(q0, q3), (q0, q4), (q0, q5),
(q1, q3), (q1, q4), (q1, q5),
(q2, q3), (q2, q4), (q2, q5)

block_plus_cross:
weak within-level + weak within-rate + stronger level-rate cross links
```

This tests whether mechanism-aligned level-rate coupling can preserve the q95 crisis advantage while repairing baseline calibration.

Success criterion:

```text
repo-style QLIKE improves materially versus RF-QRC ring
and
q95 crisis-like regime F1 remains above roughly 0.35
```

## Artifacts to preserve

Expected result tables:

- `archive/phase2/results/phase2_qrc_final_encoding_readout_predictions.csv`
- `archive/phase2/results/phase2_qrc_final_encoding_readout_prediction_export_metrics.csv`
- `phase3_rf_qrc_tail_probe_predictions_level_rate.csv` (archived off-repo in `generated_table_exports_202607.tar.gz`)
- `results/qrc/rf_qrc/phase3_rf_qrc_tail_probe_metrics_level_rate.csv`
- `results/qrc/rf_qrc/phase3_rf_qrc_tail_probe_test_summary_level_rate.csv`
- `phase3_rf_qrc_ring_leak_alpha_sweep_predictions.csv` (archived off-repo in `generated_table_exports_202607.tar.gz`)
- `results/qrc/rf_qrc/phase3_rf_qrc_ring_leak_alpha_sweep_metrics.csv`
- `results/qrc/rf_qrc/phase3_rf_qrc_ring_leak_alpha_sweep_metrics_FIXED_QLIKE.csv`, if recomputed

Expected figures from the clean notebooks:

- `results/figures/phase3_clean_actual_vs_forecasts.png`
- `results/figures/phase3_clean_raw_forecast_diagnostics.png`
- `results/figures/phase3_clean_regime_layer_f1.png`
- `results/figures/phase3_rf_qrc_ring_sweep_crisis_q95_f1_heatmap.png`
- `results/figures/phase3_rf_qrc_ring_sweep_warning_q90_f1_heatmap.png`
- `results/figures/phase3_rf_qrc_ring_sweep_qlike_heatmap.png`
- `results/figures/phase3_rf_qrc_ring_sweep_default_vs_best.png`

Before final reporting, confirm that the sweep QLIKE heatmap has been regenerated with repo-style variance QLIKE. Otherwise label the old QLIKE heatmap as wrong-scale and do not use it.

If figures are missing or were written into `notebooks/results`, regenerate them from the clean notebooks after confirming that the notebook working directory is the repository root.

## Git hygiene recommendation

Keep:

- Clean comparison notebook.
- Clean RF-QRC ring parameter sweep notebook, after fixing QLIKE.
- Scripted RF-QRC probe.
- Result tables needed to reproduce the figures.
- This findings note.

Discard or quarantine:

- Exploratory notebooks with many failed cells.
- Any `notebooks/results/` directory.
- Sweep figures/tables using the wrong QLIKE unless clearly labeled as wrong-scale.

Suggested cleanup path:

```bash
git status
rm -rf notebooks/results
mkdir -p notebooks/archive
# move messy exploratory notebooks into notebooks/archive/ only if they contain useful history
# otherwise delete them before committing

git add docs/phase3_rf_qrc_findings.md scripts results archive/phase2/results results/figures notebooks
git status
git commit -m "Document Phase 3 RF-QRC crisis-detection findings"
```

Do not commit large or redundant notebook outputs unless they are needed for reproducibility.
