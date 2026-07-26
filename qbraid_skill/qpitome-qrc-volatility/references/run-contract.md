# qBraid Financial-Classical Run Contract

## Purpose

This contract defines the executable qBraid workflow for the verified transition-data pipeline and the complete frozen classical comparison.

## Agent-owned environment setup

The judge does not create or activate an environment manually. The qBraid agent executes:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json
```

Bootstrap creates or reuses `.venv`, installs `.[test]`, runs the existing data preflight, runs the classical preflight, and executes the focused contract suite. Subsequent commands use `.venv/bin/python` explicitly.

## Data authority

The transition workflow uses the public Kaggle dataset:

```text
guillemservera/global-stock-indices-historical-data
```

The accepted source policy remains:

1. the committed fallback is usable only after complete manifest, file-set, hash, and schema verification;
2. `auto` compares an anonymous live candidate with the fallback and substitutes the fallback before installation when any file differs;
3. without a fallback, a live candidate must match the frozen raw inventory exactly;
4. the installed source is verified again;
5. the acquisition manifest records candidate comparison, installed-source comparison, fallback substitution, and substitution reason.

Run strict source preflight:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
```

Use only the reported `auto`, `fallback`, or `live` mode. Exit code 2 is a data/provenance blocker.

## Frozen classical authority

The submission models and parameters are read only from:

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
```

The canonical workflow does not tune models or evaluate test rows. The frozen models are:

```text
persistence
har
sequence_ridge
garch_1_1_t
esn_direct_tuned
esn_shuffled_tuned
```

GARCH must use backend `arch`. The development SciPy fallback is not accepted for the submitted run.

## Canonical command

Generate a literal run ID:

```bash
date -u +qbraid-financial-classical-%Y%m%dT%H%M%SZ
```

Execute:

```bash
.venv/bin/python scripts/runs/run_submission.py financial-classical \
  --run-id <RUN_ID> \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

Do not add `--force` for a new run.

## Command order

The runner executes exactly these topic stages and stops on the first nonzero return code:

1. acquire or restore verified OHLC inputs;
2. build the one-channel transition dataset;
3. build eight rematched purged walk-forward folds;
4. validate the data pipeline;
5. freeze data checksums;
6. run persistence, HAR, and sequence ridge;
7. run Student-t GARCH(1,1);
8. run the frozen tuned direct ESN and shuffled control;
9. build the exact-common-row canonical comparison;
10. validate the complete classical result family.

## Aggregate output

```text
results/runs/<RUN_ID>/
    run_manifest.json
    logs/
    files/transition_forecasting/
```

Data products remain under:

```text
files/transition_forecasting/raw/
files/transition_forecasting/processed/global_transition_dataset_1d/
files/transition_forecasting/validation/
```

Classical products are grouped by topic:

```text
files/transition_forecasting/modeling/classical_baselines/
    linear/run/<RUN_ID>/
    garch/run/<RUN_ID>/
    esn/run/<RUN_ID>/
    canonical/run/<RUN_ID>/
```

Every individual model run contains at least:

```text
params.json
config.json
dataset_manifest.json
predictions.csv.gz
submission_metrics.csv
metrics_by_fold.csv
metrics_by_horizon.csv
runtime.json
summary.json
```

The ESN run also contains `selected_spec.json`. The GARCH run also contains `fit_diagnostics.csv.gz`. The canonical run contains:

```text
common_predictions.csv.gz
coverage.csv
paired_deltas_vs_sequence_ridge.csv
submission_table_selection.csv
submission_table_confirmation.csv
submission_table_development_all.csv
```

The validation topic must contain:

```text
files/transition_forecasting/validation/classical_baseline_audit.json
```

## Reporting contract

For every model, the canonical tables report:

```text
Transition
L1
L5
L10
Controls
Pooled
```

Controls are never subdivided by lead. Metrics are:

```text
RMSE
log-volatility QLIKE
Mincer-Zarnowitz alpha
Mincer-Zarnowitz beta
Mincer-Zarnowitz R2
```

The canonical comparison scores only the exact common finite `(fold, sample_id)` intersection across all six models.

## Acceptance conditions

A run is accepted only when:

1. `run_manifest.json` records `status: succeeded` and `workflow: financial-classical`;
2. every required data and classical output exists and has a SHA-256 value;
3. the raw acquisition manifest verifies the installed source against the authoritative reference;
4. data audit and checksum reports pass;
5. `classical_baseline_audit.json` records `passed: true`;
6. the data identity remains one channel, 40 input sessions, 10 target sessions, and eight folds;
7. the control protocol remains `precontrol_binary_matching`;
8. predictions contain validation rows only from folds 4–8;
9. the fixed test partition remains unevaluated;
10. the canonical model and group sets exactly match this contract;
11. no control-by-lead row exists;
12. GARCH records backend `arch`;
13. ESN parameters match the frozen specification;
14. no `.py` file exists beneath the aggregate run directory;
15. every command, log, and artifact belongs to the same explicit run ID.

## Hardware boundary

This workflow must not query, select, or submit to a QPU. Financial QRC and other challenge stages will be added separately.

## Failure handling

On any failure, preserve the run directory and logs, report the first failed command and exact log path, classify the blocker, and do not patch scientific behavior during reproduction.
