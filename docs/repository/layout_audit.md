# Repository layout audit

Reorganization applied: **False**

## Modeling moves

- `scripts/modeling/transition_destination/analyze_branch_destination_rydberg_increment.py` -> `scripts/modeling/task_b_destination/analyze_branch_destination_rydberg_increment.py`
- `scripts/modeling/transition_destination/analyze_branch_destination_structure.py` -> `scripts/modeling/task_b_destination/analyze_branch_destination_structure.py`
- `scripts/modeling/transition_onset/analyze_branch_onset_trajectory.py` -> `scripts/modeling/task_a_onset/analyze_branch_onset_trajectory.py`
- `scripts/modeling/day5_branching/baseline/analyze_cross_market_day5_direction_robustness.py` -> `scripts/modeling/day5_barrier/analyze_cross_market_day5_direction_robustness.py`
- `scripts/modeling/weekly_regimes/analyze_weekly_regime_baselines.py` -> `scripts/modeling/weekly_regimes/analyze_weekly_regime_baselines.py`
- `scripts/modeling/weekly_regimes/analyze_weekly_regime_states.py` -> `scripts/modeling/weekly_regimes/analyze_weekly_regime_states.py`
- `scripts/modeling/transition_destination/audit_branch_destination_targets.py` -> `scripts/modeling/task_b_destination/audit_branch_destination_targets.py`
- `scripts/modeling/transition_onset/audit_branch_onset_triviality.py` -> `scripts/modeling/task_a_onset/audit_branch_onset_triviality.py`
- `scripts/modeling/day5_branching/baseline/build_cross_market_day5_path_panel_exploratory.py` -> `scripts/modeling/day5_barrier/build_cross_market_day5_path_panel_exploratory.py`
- `scripts/modeling/day5_branching/residual_confirmation/merge_day5_residualized_rydberg.py` -> `scripts/modeling/day5_barrier/merge_day5_residualized_rydberg.py`
- `scripts/modeling/day5_branching/residual_confirmation/merge_day5_residualized_rydberg_crossfit.py` -> `scripts/modeling/day5_barrier/merge_day5_residualized_rydberg_crossfit.py`
- `scripts/modeling/day5_branching/spatial_rydberg/merge_day5_spatial_rydberg_assay.py` -> `scripts/modeling/day5_barrier/merge_day5_spatial_rydberg_assay.py`
- `scripts/modeling/transition_destination/prepare_branch_destination_paths.py` -> `scripts/modeling/task_b_destination/prepare_branch_destination_paths.py`
- `scripts/modeling/transition_destination/prepare_branch_destination_rydberg_inputs.py` -> `scripts/modeling/task_b_destination/prepare_branch_destination_rydberg_inputs.py`
- `scripts/modeling/transition_destination/run_branch_destination_classical_baselines.py` -> `scripts/modeling/task_b_destination/run_branch_destination_classical_baselines.py`
- `scripts/modeling/transition_destination/run_branch_destination_esn_baseline.py` -> `scripts/modeling/task_b_destination/run_branch_destination_esn_baseline.py`
- `scripts/modeling/transition_destination/run_branch_destination_rydberg_simulator.py` -> `scripts/modeling/task_b_destination/run_branch_destination_rydberg_simulator.py`
- `scripts/modeling/day5_branching/baseline/run_cross_market_day5_confirmatory_baselines.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_confirmatory_baselines.py`
- `scripts/modeling/day5_branching/static_rydberg/run_cross_market_day5_differential_local_rydberg_probe.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_differential_local_rydberg_probe.py`
- `scripts/modeling/day5_branching/baseline/run_cross_market_day5_direction_baselines.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_direction_baselines.py`
- `scripts/modeling/day5_branching/temporal_controls/run_cross_market_day5_esn_probe.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_esn_probe.py`
- `scripts/modeling/day5_branching/temporal_controls/run_cross_market_day5_fixed_esn.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_fixed_esn.py`
- `scripts/modeling/day5_branching/static_rydberg/run_cross_market_day5_fixed_quantum_feature.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_fixed_quantum_feature.py`
- `scripts/modeling/day5_branching/static_rydberg/run_cross_market_day5_fixed_rydberg_feature.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_fixed_rydberg_feature.py`
- `scripts/modeling/day5_branching/baseline/run_cross_market_day5_regularized_controls.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_regularized_controls.py`
- `scripts/modeling/day5_branching/static_rydberg/run_cross_market_day5_static_nonlinear_probe.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_static_nonlinear_probe.py`
- `scripts/modeling/day5_branching/static_rydberg/run_cross_market_day5_static_rydberg_probe.py` -> `scripts/modeling/day5_barrier/run_cross_market_day5_static_rydberg_probe.py`
- `scripts/modeling/day5_branching/static_rydberg/run_day5_differential_local_rydberg_diagnostic.py` -> `scripts/modeling/day5_barrier/run_day5_differential_local_rydberg_diagnostic.py`
- `scripts/modeling/day5_branching/falsification/run_day5_first_passage_sanity.py` -> `scripts/modeling/day5_barrier/run_day5_first_passage_sanity.py`
- `scripts/modeling/day5_branching/residual_confirmation/run_day5_higher_order_output_shard.py` -> `scripts/modeling/day5_barrier/run_day5_higher_order_output_shard.py`
- `scripts/modeling/day5_branching/falsification/run_day5_input_audit.py` -> `scripts/modeling/day5_barrier/run_day5_input_audit.py`
- `scripts/modeling/day5_branching/residual_confirmation/run_day5_nonlinear_input_residual.py` -> `scripts/modeling/day5_barrier/run_day5_nonlinear_input_residual.py`
- `scripts/modeling/day5_branching/residual_confirmation/run_day5_protected_input_residual.py` -> `scripts/modeling/day5_barrier/run_day5_protected_input_residual.py`
- `scripts/modeling/day5_branching/residual_confirmation/run_day5_residualized_rydberg_crossfit_shard.py` -> `scripts/modeling/day5_barrier/run_day5_residualized_rydberg_crossfit_shard.py`
- `scripts/modeling/day5_branching/residual_confirmation/run_day5_residualized_rydberg_shard.py` -> `scripts/modeling/day5_barrier/run_day5_residualized_rydberg_shard.py`
- `scripts/modeling/day5_branching/spatial_rydberg/run_day5_rydberg_pca_reconstruction.py` -> `scripts/modeling/day5_barrier/run_day5_rydberg_pca_reconstruction.py`
- `scripts/modeling/day5_branching/spatial_rydberg/run_day5_spatial_rydberg_assay_shard.py` -> `scripts/modeling/day5_barrier/run_day5_spatial_rydberg_assay_shard.py`
- `scripts/modeling/day5_branching/residual_confirmation/run_day5_standard_feature_assay.py` -> `scripts/modeling/day5_barrier/run_day5_standard_feature_assay.py`
- `scripts/modeling/weekly_regimes/run_weekly_regime_baselines.py` -> `scripts/modeling/weekly_regimes/run_weekly_regime_baselines.py`

## Markdown references updated

- `None`

## Remaining layout findings

### Top Level Scripts

- None

### Flat Modeling Scripts

- `analyze_branch_destination_rydberg_increment.py`
- `analyze_branch_destination_structure.py`
- `analyze_branch_onset_trajectory.py`
- `analyze_cross_market_day5_direction_robustness.py`
- `analyze_weekly_regime_baselines.py`
- `analyze_weekly_regime_states.py`
- `audit_branch_destination_targets.py`
- `audit_branch_onset_triviality.py`
- `build_cross_market_day5_path_panel_exploratory.py`
- `merge_day5_residualized_rydberg.py`
- `merge_day5_residualized_rydberg_crossfit.py`
- `merge_day5_spatial_rydberg_assay.py`
- `prepare_branch_destination_paths.py`
- `prepare_branch_destination_rydberg_inputs.py`
- `run_branch_destination_classical_baselines.py`
- `run_branch_destination_esn_baseline.py`
- `run_branch_destination_rydberg_simulator.py`
- `run_cross_market_day5_confirmatory_baselines.py`
- `run_cross_market_day5_differential_local_rydberg_probe.py`
- `run_cross_market_day5_direction_baselines.py`
- `run_cross_market_day5_esn_probe.py`
- `run_cross_market_day5_fixed_esn.py`
- `run_cross_market_day5_fixed_quantum_feature.py`
- `run_cross_market_day5_fixed_rydberg_feature.py`
- `run_cross_market_day5_regularized_controls.py`
- `run_cross_market_day5_static_nonlinear_probe.py`
- `run_cross_market_day5_static_rydberg_probe.py`
- `run_day5_differential_local_rydberg_diagnostic.py`
- `run_day5_first_passage_sanity.py`
- `run_day5_higher_order_output_shard.py`
- `run_day5_input_audit.py`
- `run_day5_nonlinear_input_residual.py`
- `run_day5_protected_input_residual.py`
- `run_day5_residualized_rydberg_crossfit_shard.py`
- `run_day5_residualized_rydberg_shard.py`
- `run_day5_rydberg_pca_reconstruction.py`
- `run_day5_spatial_rydberg_assay_shard.py`
- `run_day5_standard_feature_assay.py`
- `run_weekly_regime_baselines.py`

### Root Data Files

- None

### Result Directories Without Manifest

- `results/modeling/day5_branching/baseline/cross_market_day5_confirmatory_v1`
- `results/modeling/day5_branching/baseline/cross_market_day5_direction_v1/robustness`
- `results/modeling/day5_branching/baseline/cross_market_day5_regularized_controls_v1`
- `results/canonical/current`
- `results/canonical/segments/esn_selected`
- `results/canonical/segments/har_ridge`
- `results/canonical/segments/persistence_20d`
- `results/canonical/segments/raw_ridge`
- `results/canonical/segments/rydberg_memoryless`
- `results/canonical/segments/rydberg_multi_lb`
- `results/canonical/segments/rydberg_multi_lb_memoryless`
- `results/canonical/segments/rydberg_multi_lb_shuffled`
- `results/canonical/segments/rydberg_shuffled`
- `results/canonical/segments/rydberg_temporal`
- `results/canonical/segments/tfim_phase2_final`
- `results/canonical/transition_v1`
- `results/canonical/transition_v1/esn`
- `results/canonical/transition_v1/garch`
- `results/canonical/transition_v1/lstm`
- `results/canonical/transition_v1/tfim`
- `results/diagnostics/cross_market_branch_portability_v1/ftse_100`
- `results/diagnostics/cross_market_branch_portability_v1/nikkei_225`
- `results/diagnostics/cross_market_branch_portability_v1/russell_2000`
- `results/diagnostics/cross_market_crisis_clusters_v2`
- `results/experiments/rydberg_temporal_memory_ablation_v1`
- `results/experiments/rydberg_two_channel_history_control_v1`
- `results/figures`
- `results/modeling/day5_branching/static_rydberg/cross_market_day5_fixed_quantum_feature_v1`
- `results/modeling/day5_branching/static_rydberg/cross_market_day5_fixed_rydberg_feature_v1`
- `results/qrc/rf_qrc`
- `results/qrc/rydberg`
- `results/qrc/rydberg/phase3_rydberg_temporal_robustness`
- `results/reference/phase2`
- `results/regimes/branch_outcome_label_audit_v1`
- `results/regimes/branching_extractor_audit_v1/episode_sets`
- `results/regimes/long_history_branch_reconstruction_v2`
- `results/runs`
