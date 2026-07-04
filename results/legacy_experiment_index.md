# Historical experiment inventory

This index classifies the restored result tables by evidentiary role. It is not a canonical leaderboard. The purpose is to preserve the experimental history while making clear what each result family can and cannot support.

## Reference results

### Phase 2 classical baselines

Files:
- `results/tables/phase2_classical_baselines.csv`

Role: classical reference under the original fixed chronological train/validation/test split.

Key findings for the 20-day volatility target:
- HAR ridge test RMSE 0.10071, QLIKE -2.07912, MZ R2 0.35370.
- HAR linear is effectively identical.
- Full-feature ridge/elastic net do not improve on HAR out of sample.
- Persistence is materially weaker.

Use: historical reference only until rerun under the common purged walk-forward evaluator.

### Phase 2 ESN

Files:
- `results/tables/phase2_esn_prediction_export_metrics.csv`
- `results/tables/phase2_esn_predictions.csv`
- `results/reference/phase2/phase2_esn_log_target_confirmation_aggregate.csv`

Role: strongest classical reservoir reference from Phase 2.

Important distinction:
- Best single-seed exports reach test RMSE about 0.07177-0.07190 and MZ R2 about 0.531-0.540.
- Five-seed confirmation is weaker and should be reported separately rather than merged into one generic "ESN baseline".

Use: historical reference. A common purged walk-forward ESN comparison is still required for a clean Phase 3 claim.

### Phase 2 TFIM/QRC

Files:
- `results/tables/phase2_feedback_tfim_qrc_reference_metrics.csv`
- `results/tables/phase2_feedback_tfim_qrc_reference_predictions.csv`
- `results/tables/phase2_qrc_final_encoding_readout_prediction_export_metrics.csv`
- `results/tables/phase2_qrc_final_encoding_readout_predictions.csv`

Role: historical quantum reference.

Key findings:
- Feedback-TFIM reference test RMSE 0.09675, QLIKE -2.29061, MZ R2 0.15075.
- Final encoded/readout QRC test RMSE 0.09508, QLIKE -2.22972, MZ R2 0.18890, correlation 0.43462.
- Phase 2 TFIM/QRC improved substantially over earlier weak QRC variants but remained below ESN on continuous forecasting.

Use: historical reference. Tail ranking under the current purged walk-forward protocol remains an open comparison.

## Early Phase 3 exploratory branches

### Clean regime-layer comparison

Files:
- `results/tables/phase3_clean_full_regime_layer_evaluation.csv`
- `results/tables/phase3_clean_main_regime_layer_comparison.csv`
- `results/tables/phase3_clean_raw_forecast_comparison.csv`
- `results/tables/phase3_clean_regime_counts.csv`

Role: exploratory interpretation layer that first showed a mismatch between regression quality and tail-warning behavior.

Key findings:
- Historical ESN remains strongest on raw continuous forecast metrics.
- RF-QRC ring showed much stronger q95 amplitude F1 than Phase 2 QRC and changed the direction of Phase 3 research toward rare-event ranking.
- Forecast-aware regime thresholds produced a q95 crisis-like F1 advantage for RF-QRC relative to the historical ESN reference in this specific fixed-split setup.

Caveat: this is not the current purged walk-forward protocol and should not be treated as canonical advantage evidence.

### RF-QRC leak/ridge sweep

Files:
- `results/tables/phase3_rf_qrc_ring_leak_alpha_sweep_metrics.csv`
- `results/tables/phase3_rf_qrc_ring_leak_alpha_sweep_metrics_FIXED_QLIKE.csv`
- `results/tables/phase3_rf_qrc_ring_leak_alpha_sweep_predictions.csv`

Role: tuning.

Key findings:
- Leak and ridge regularization strongly changed amplitude calibration and tail F1.
- The best crisis-like F1 in the sweep was obtained around leak 0.3 with moderate ridge, but continuous RMSE/QLIKE remained poor.
- Strong sensitivity to regularization means these results are unsuitable as a standalone benchmark without strict validation-only selection.

### RF-QRC tail probe

Files:
- `results/tables/phase3_rf_qrc_tail_probe_metrics_level_rate.csv`
- `results/tables/phase3_rf_qrc_tail_probe_predictions_level_rate.csv`
- `results/tables/phase3_rf_qrc_tail_probe_test_summary_level_rate.csv`

Role: exploratory architecture probe.

Key findings:
- Adding a second encode and ring entanglement increased correlation and tail F1 relative to simpler variants.
- The ring variant reached q95 F1 0.32 in the fixed-split test, but continuous forecast quality remained weak.

### Time multiplexing

Files:
- `results/tables/phase3_rf_qrc_time_multiplex_metrics.csv`
- `results/tables/phase3_rf_qrc_time_multiplex_predictions.csv`
- `results/tables/phase3_rf_qrc_time_multiplex_selection_table.csv`
- `results/tables/phase3_rf_qrc_time_multiplex_test_summary.csv`

Role: tuning/experimental.

Key findings:
- Increasing virtual nodes increased effective rank.
- Some q95 F1 values improved, with the strongest listed result around 5 virtual nodes and ridge 1000.
- No configuration passed the experiment's predefined time-multiplexing gate.

Conclusion: useful negative result. Higher feature rank did not establish a robust benefit sufficient to justify additional complexity.

### One-QRC/two-head readout

Files:
- `results/tables/phase3_one_qrc_two_head_classifier_metrics.csv`
- `results/tables/phase3_one_qrc_two_head_predictions.csv`
- `results/tables/phase3_one_qrc_two_head_regression_metrics.csv`
- `results/tables/phase3_one_qrc_two_head_test_summary.csv`

Role: experimental readout study.

Key findings:
- A separate logistic tail head improved q80/q90 ranking behavior relative to the ridge regression amplitude thresholding.
- q95 validation labels were degenerate in this fixed split, causing threshold selection failure and an almost-always-positive test classifier.

Conclusion: direct classifier heads can help, but this experiment cannot support a q95 claim because the validation protocol was degenerate.

### Structured level/rate RF-QRC

Files:
- `results/tables/phase3_structured_level_rate_rf_qrc_metrics.csv`
- `results/tables/phase3_structured_level_rate_rf_qrc_predictions.csv`
- `results/tables/phase3_structured_level_rate_rf_qrc_selection_table.csv`
- `results/tables/phase3_structured_level_rate_rf_qrc_test_summary.csv`

Role: architecture ablation.

Key findings:
- The ring control clearly outperformed cross-matched, cross-all, and block-plus-cross entangling structures on q95 F1.
- No structured alternative passed the predefined gate.

Conclusion: retain as a negative ablation supporting the ring choice in the historical RF-QRC branch.

## Diagnostics and scaling

### Qubit-scaling transition classifier

Files:
- `results/tables/phase3_qubit_scaling_q95_focus.csv`
- `results/tables/phase3_qubit_scaling_transition_classifier_metrics.csv`
- `results/tables/phase3_qubit_scaling_transition_classifier_predictions.csv`
- `results/tables/phase3_qubit_scaling_transition_classifier_test_summary.csv`

Role: diagnostic/scaling study.

Key findings:
- QRC-only performance was not monotonic in qubit count.
- The 8-qubit hybrid HAR+QRC model produced the best listed q95 AP (0.42262), modestly above HAR alone (0.39923), while 10-qubit QRC-only degraded.
- Feature-generation time rose sharply: about 6.5 s at 6 qubits, 22.4 s at 8 qubits, 97.3 s at 10 qubits for the recorded study.

Conclusion: more qubits did not imply more value. The hybrid signal at 8 qubits was an early hint that quantum features may be conditionally useful as an augmentation rather than a universal replacement.

### TFIM N/H scaling

Files:
- `results/tables/phase3_tfim_n_h_scaling_metrics.csv`
- `results/tables/phase3_tfim_n_h_scaling_predictions.csv`
- `results/tables/phase3_tfim_n_h_scaling_test_agg.csv`
- `results/tables/phase3_tfim_n_h_scaling_test_summary.csv`

Role: diagnostic/scaling study.

Key findings:
- Increasing qubit count raised effective rank and compute cost dramatically.
- Tail F1 remained mostly near zero; the best average q95 F1 was still weak.
- Runtime rose from seconds at 4 qubits to thousands of seconds at 12 qubits.

Conclusion: a strong negative scaling result. Larger TFIM state spaces did not create useful forecast capacity proportional to cost.

## Rydberg temporal robustness sweep

Directory:
- `results/tables/phase3_rydberg_temporal_robustness/`

Role: large exploratory/tuning sweep, not canonical evidence.

Design:
- scalar inputs: `rv_accel_log_5_20`, `vix_rv_spread`
- lookbacks: 10, 20, 40
- three detuning windows
- 18 configurations total
- raw baseline, Rydberg temporal features, and raw-plus-Rydberg comparisons

Key aggregate findings on test data:
- q90 AP improved in 2/18 configurations.
- q95 AP improved in 2/18 configurations.
- q95 AUC improved in 12/18 configurations.
- regression RMSE improved in 5/18 configurations.

Interpretation:
- The branch frequently changed ranking geometry without improving precision-recall performance.
- It is an important failed/partial branch because it shows that generic Rydberg temporal processing was not sufficient.
- It should be retained for provenance and mechanism history, but not mixed with the later true temporal dual-chain Rydberg result.

## Current Phase 3 results not represented by this restored legacy table set

The most important current results still live primarily in git-ignored scratch outputs or later scripts and therefore are not yet represented here as stable committed result tables:

- true temporal dual-chain Rydberg purged walk-forward comparison
- memoryless and shuffled controls under the same current evaluator
- current AP-push / multi-lookback results
- true multinomial bitstring finite-shot degradation
- current Phase 3 ESN purged walk-forward workup
- hardware MVP window selection and eventual QPU outputs

These should be the first results promoted into the new structured result directories.

## Classification rule going forward

- `results/reference/` — historical baselines used for comparison
- `results/canonical/` — common-protocol model comparisons only
- `results/tuning/` — parameter sweeps and selection tables
- `results/experimental/` — architecture and readout probes
- `results/diagnostics/` — mechanism, scaling, and ablation studies
- `results/hardware/` — finite-shot and real-device studies
- `results/archive/` — superseded or failed branches retained for provenance

The existing `results/tables/` directory should be treated as a legacy evidence store. Do not add new experiments there.
