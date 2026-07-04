# Tuning results

Parameter searches and model-selection studies. These results document what was searched; they are not canonical benchmark evidence by themselves.

## Historical RF-QRC leak/ridge sweep

- `../tables/phase3_rf_qrc_ring_leak_alpha_sweep_metrics.csv`
- `../tables/phase3_rf_qrc_ring_leak_alpha_sweep_metrics_FIXED_QLIKE.csv`
- `../tables/phase3_rf_qrc_ring_leak_alpha_sweep_predictions.csv`

## Historical RF-QRC time multiplexing

- `../tables/phase3_rf_qrc_time_multiplex_metrics.csv`
- `../tables/phase3_rf_qrc_time_multiplex_selection_table.csv`
- `../tables/phase3_rf_qrc_time_multiplex_test_summary.csv`
- `../tables/phase3_rf_qrc_time_multiplex_predictions.csv`

No tested time-multiplex configuration passed the original gate.

## Historical Rydberg temporal robustness sweep

- `../tables/phase3_rydberg_temporal_robustness/`

Design: 2 scalar inputs x 3 lookbacks x 3 detuning windows = 18 configurations.

Aggregate result: q95 AP improved over the raw baseline in only 2/18 test configurations; q95 AUC improved in 12/18. Retain as a failed/partial search branch, not as current Rydberg evidence.

Future parameter-search runners should write to `results/tuning/<search_name>/` with explicit search configuration, trial table, selected configuration, and selection metric.
