# qBraid Stage-1 Run Contract

## Purpose

This contract defines the currently testable qBraid execution path. It intentionally covers only functionality present on `stage1-dev`.

## Agent-owned environment setup

The judge should not create or activate an environment manually. The qBraid agent runs:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json
```

The bootstrap:

- creates the repository-local `.venv` if absent;
- installs the project and test dependencies;
- runs preflight;
- runs the focused Stage-1 and skill tests;
- stops on the first failure.

Subsequent commands must use `.venv/bin/python` or `.venv/bin/kaggle` explicitly. Do not use a bare `pip`, system-wide installation, or rely on shell activation persisting between agent actions.

## Data dependency

The transition workflow uses the public Kaggle dataset:

```text
guillemservera/global-stock-indices-historical-data
```

The current branch does not contain the large verified fallback snapshot. A clean qBraid clone therefore cannot complete the data run unless secure Kaggle access already exists. Never commit, print, request in chat, or copy credentials into the repository or result directory.

The final submission should prefer a committed credential-free verified fallback.

## Preflight

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --json
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
```

Exit codes:

- `0`: repository contract passes; strict mode also found a usable source;
- `1`: Python, dependency, or repository contract failure;
- `2`: strict mode found neither secure Kaggle access nor the verified fallback.

Exit code 2 is a data/provenance blocker, not permission to improvise another dataset.

## Canonical command

Generate a literal run ID:

```bash
date -u +qbraid-stage1-%Y%m%dT%H%M%SZ
```

With the verified fallback:

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \
  --run-id <RUN_ID> \
  --transition-source-mode fallback
```

With verified live access:

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \
  --run-id <RUN_ID> \
  --transition-source-mode live
```

Do not use `auto` while source availability is ambiguous. Do not use `--force` for a new run.

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
7. the protocol is `precontrol_binary_matching`;
8. no `.py` file exists beneath the run directory;
9. all commands, logs, and artifacts correspond to the same explicit run ID.

## Hardware boundary

This run must not query, select, or submit to a QPU. Fresh hardware execution is outside this workflow.

## Current limitations

This Stage-1 contract does not yet reproduce:

- the final financial HAR/QRC comparison;
- final QRC training or inference;
- MNIST;
- qubit scaling;
- noise analysis;
- final report tables and figures;
- final judge-facing artifact collection.

Those stages must be added to the single automated runner before the Phase 3 submission skill is end-to-end.
