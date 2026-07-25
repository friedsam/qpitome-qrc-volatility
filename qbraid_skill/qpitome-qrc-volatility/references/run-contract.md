# qBraid Stage-1 Run Contract

## Purpose

This contract defines the currently testable qBraid execution path on `stage1-dev`.

## Agent-owned environment setup

The judge should not create or activate an environment manually. The qBraid agent runs:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json
```

The bootstrap creates `.venv` if absent, installs dependencies, runs preflight, runs the focused Stage-1 and skill tests, and stops on the first failure. Subsequent commands use `.venv/bin/python` or `.venv/bin/kaggle` explicitly.

## Data dependency and authority

The transition workflow uses the public Kaggle dataset:

```text
guillemservera/global-stock-indices-historical-data
```

Current Kaggle CLI releases permit anonymous `datasets files` and `datasets download` for public datasets. Preflight therefore tests the endpoint directly rather than treating credentials as mandatory.

The authoritative data policy is:

1. A fallback is usable only after its manifest, complete file set, hashes, and schemas verify.
2. When anonymous live access and the fallback are both available, `auto` downloads a live candidate and compares its complete data-file inventory to the fallback.
3. If the candidate differs by any missing, extra, or changed file, it is discarded before installation and the verified fallback is copied instead.
4. When no fallback exists, a live candidate must match the frozen raw inventory exactly or acquisition fails.
5. The installed destination is verified again after copying or download.

The acquisition manifest records the candidate comparison, installed-source comparison, fallback substitution flag, and substitution reason.

## Preflight

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --json
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
```

Exit codes:

- `0`: repository contract passes; strict mode found anonymous live access or a verified fallback;
- `1`: Python, dependency, repository, or fallback-verification failure;
- `2`: strict mode found no verified data source.

Exit code 2 is a data/provenance blocker, not permission to improvise another dataset.

## Canonical command

Generate a literal run ID:

```bash
date -u +qbraid-stage1-%Y%m%dT%H%M%SZ
```

Use the exact mode reported by preflight. When both sources are ready:

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \
  --run-id <RUN_ID> \
  --transition-source-mode auto
```

Use `fallback` or `live` only when preflight reports that as the sole verified mode. Do not use `--force` for a new run.

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

1. `run_manifest.json` records `status: succeeded`;
2. every required output exists and has a recorded SHA-256 hash;
3. `raw_acquisition_manifest.json` records `authoritative_source_verified: true`;
4. `installed_source_comparison.matched` is true;
5. any live mismatch or failure and fallback substitution are explicitly recorded;
6. the data audit records `passed: true`;
7. the checksum report records `passed: true`;
8. the test partition remains unevaluated;
9. the workflow records eight folds and one `log_volatility_level` channel;
10. the protocol is `precontrol_binary_matching`;
11. no `.py` file exists beneath the run directory;
12. all commands, logs, and artifacts correspond to the same explicit run ID.

## Hardware boundary

This run must not query, select, or submit to a QPU. Fresh hardware execution is outside this workflow.

## Current limitations

This Stage-1 contract does not yet reproduce the final financial HAR/QRC comparison, final QRC training or inference, MNIST, qubit scaling, noise analysis, final report tables and figures, or final judge-facing artifact collection. Those stages must be added to the single automated runner before submission.
