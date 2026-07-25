# qBraid Stage-1 Run Contract

## Purpose

This contract defines the currently testable qBraid execution path. It intentionally describes only functionality present on `stage1-dev`.

## Environment

- Python 3.10 or newer; the project environment currently targets Python 3.11.
- Install through the active interpreter:

```bash
python -m pip install -e ".[test]"
```

Do not use a bare `pip` command on qBraid. qBraid environments can have multiple Python installations, and installing into the wrong interpreter can produce a nonpersistent or inconsistent environment.

## Data dependency

The transition workflow uses the public Kaggle dataset:

```text
guillemservera/global-stock-indices-historical-data
```

The current branch does not contain the large verified fallback snapshot. A clean qBraid clone therefore requires:

- the Kaggle CLI installed by the project dependencies;
- a valid `~/.kaggle/access_token` or `~/.kaggle/kaggle.json`;
- restrictive credential-file permissions;
- dataset access verified before execution.

Never commit, print, or copy credentials into the repository or result directory.

## Preflight

```bash
python qbraid_skill/scripts/preflight.py --json
python qbraid_skill/scripts/preflight.py --strict-data-source
```

Exit codes:

- `0`: repository contract passes; strict mode also has a usable data source;
- `1`: Python, dependency, or repository contract failure;
- `2`: strict mode found neither secure Kaggle access nor the verified fallback.

## Canonical command

```bash
RUN_ID="qbraid-stage1-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/runs/run_submission.py transition-data \
  --run-id "$RUN_ID" \
  --transition-source-mode live
```

Use `fallback` only when the expected fallback manifest and its complete verified snapshot are present. Do not use `auto` on the current clean-clone branch because the absent fallback makes its second path nonfunctional.

## Expected top-level run files

```text
results/runs/<run-id>/run_manifest.json
results/runs/<run-id>/logs/
results/runs/<run-id>/files/transition_forecasting/
```

The transition subtree must include:

```text
raw/global_stock_indices_historical_data/all_indices_data.csv
raw/global_stock_indices_historical_data/raw_acquisition_manifest.json
processed/global_transition_dataset_1d/cleaned_ohlc.csv.gz
processed/global_transition_dataset_1d/daily_volatility.csv.gz
processed/global_transition_dataset_1d/transition_catalogue.csv
processed/global_transition_dataset_1d/sample_manifest.csv
processed/global_transition_dataset_1d/sequence_tensors.npz
processed/global_transition_dataset_1d/row_corrections.csv
processed/global_transition_dataset_1d/manifest.json
processed/global_transition_dataset_1d/control_candidate_manifest.csv
processed/global_transition_dataset_1d/control_candidate_tensors.npz
processed/global_transition_dataset_1d/candidate_pool_summary.json
processed/global_transition_dataset_1d/purged_walk_forward_folds/rematched_rolling_manifest.csv
processed/global_transition_dataset_1d/purged_walk_forward_folds/rematched_rolling_tensors.npz
processed/global_transition_dataset_1d/purged_walk_forward_folds/control_match_audit.csv
processed/global_transition_dataset_1d/purged_walk_forward_folds/summary.json
validation/data_pipeline_audit.json
validation/data_pipeline_checksums.json
```

## Acceptance conditions

A run is accepted only when:

1. `run_manifest.json` records `status` as `succeeded`;
2. every required output exists and has a recorded SHA-256 hash;
3. the data audit records `passed: true`;
4. the checksum report records `passed: true`;
5. the test partition remains unevaluated;
6. the workflow records eight folds and one `log_volatility_level` channel;
7. no `.py` file exists under the run directory;
8. all commands and logs correspond to the same explicit run ID.

## Hardware boundary

This run must not query, select, or submit to a QPU. qBraid Agent Mode requires approval for sensitive hardware execution, but this repository contract is stricter: fresh hardware execution is outside this workflow entirely.

## Current limitations

This Stage-1 contract does not yet reproduce:

- the final financial HAR/QRC comparison;
- final QRC training or inference;
- MNIST;
- qubit scaling;
- noise analysis;
- final report tables and figures;
- final judge-facing artifact collection.

Those stages must be added to the single automated runner before the submission skill can be considered end-to-end for Phase 3.