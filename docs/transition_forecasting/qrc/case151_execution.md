# Case151 execution in the submission repository

Run the tested canonical simulator stage with the same aggregate run ID used by the
submission pipeline:

```bash
python scripts/runs/run_case151_submission.py \
  --run-id <RUN_ID> \
  --fold-dir results/runs/<RUN_ID>/files/data/processed/global_transition_dataset_1d/purged_walk_forward_folds
```

The stage writes:

```text
results/runs/<RUN_ID>/
  logs/qrc_case151.log
  files/qrc/simulation/run/<RUN_ID>/
    params.json
    prediction_cells.csv.gz
    fold_metrics.csv
    pooled_metrics.csv
    readout_selections.csv
    readout_candidates.csv.gz
    feature_diagnostics.csv
    baseline_health.csv
    channel_scalers.csv
    simulation_metadata.csv
    summary.json
    artifact_manifest.json
```

The exact model is the six-atom A/4-B/2-A/4 palindrome using 63 occupation-and-pair
features and fold-local chronological alpha/lambda selection. Case151 is the fold-8
instance with alpha 0.1 and lambda 0.25.

The hardware utilities support only `prepare`, `status`, and `collect` for the three
already-completed Aquila jobs. They do not expose a hardware-submission command.
