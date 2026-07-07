# Diagnostic results

This area is for mechanism, scaling, geometry, memory, and ablation studies.

Historical diagnostic files remain in `results/tables/`:

- `phase3_qubit_scaling_q95_focus.csv`
- `phase3_qubit_scaling_transition_classifier_metrics.csv`
- `phase3_qubit_scaling_transition_classifier_test_summary.csv`
- `phase3_qubit_scaling_transition_classifier_predictions.csv`
- `phase3_tfim_n_h_scaling_metrics.csv`
- `phase3_tfim_n_h_scaling_test_agg.csv`
- `phase3_tfim_n_h_scaling_test_summary.csv`
- `phase3_tfim_n_h_scaling_predictions.csv`

The qubit-scaling study showed non-monotonic model quality with sharply increasing compute time. The TFIM scaling study showed increasing effective rank and runtime without proportional forecasting improvement.

Stable dual-chain and temporal-memory diagnostics should be promoted here.
