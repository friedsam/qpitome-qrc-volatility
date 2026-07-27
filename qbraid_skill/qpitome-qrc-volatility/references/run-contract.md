# qBraid Core and Full-Smoke Reproduction Contract

## Purpose

This contract defines the executable qBraid workflow for the verified transition-data pipeline, complete frozen classical comparison, canonical Case151 QRC simulation, and optional bounded Phase-3 smoke studies.

The default **core** scope is:

```text
data + classical baselines + canonical Case151 QRC
```

The optional **full-smoke** scope is:

```text
core + MNIST smoke + noise smoke + scaling smoke + finite-shot smoke
```

Hardware is never required by either scope.

## Agent-owned environment setup

The judge does not create or activate an environment manually. The qBraid agent executes:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json
```

Bootstrap creates or reuses `.venv`, installs `.[test]`, runs the data and classical preflights, and executes focused data, classical, Case151, benchmark, and hardware-safety contract tests. Subsequent commands use `.venv/bin/python` explicitly.

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

## Frozen Case151 authority

The canonical financial QRC configuration and metric oracle are fixed by:

```text
scripts/reproduction/run_case151_simulation.py
config/case151/expected_metrics.json
config/case151/agent_run_spec.json
```

The accepted identity is:

- exact six-atom simulator;
- A/4 → B/2 → A/4 palindrome;
- three probe times;
- `occupation_pair_raw` feature bank;
- 63 occupation/pair features;
- fold-specific chronological alpha/lambda selection;
- fold-8 alpha `0.1` and lambda `0.25`;
- intercept-free residual head;
- zero fixed-test rows.

Do not substitute another QRC readout or later research candidate.

## Core commands

Generate one literal run ID:

```bash
date -u +qbraid-financial-classical-%Y%m%dT%H%M%SZ
```

Run data and classical stages:

```bash
.venv/bin/python scripts/runs/run_submission_stage_layout.py financial-classical \
  --run-id <RUN_ID> \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

Do not add `--force` for a new run.

After accepting the data/classical stage, run Case151 against the folds in the same aggregate directory:

```bash
.venv/bin/python scripts/reproduction/run_case151_simulation.py \
  --fold-dir results/runs/<RUN_ID>/files/data/processed/global_transition_dataset_1d/purged_walk_forward_folds \
  --output-root results/runs/<RUN_ID>/files/qrc/simulation/run \
  --run-id <RUN_ID>
```

## Core command order

1. acquire or restore verified OHLC inputs;
2. build the one-channel transition dataset;
3. build eight rematched purged walk-forward folds;
4. validate the data pipeline;
5. freeze data checksums;
6. run persistence, HAR, and sequence ridge;
7. run Student-t GARCH(1,1);
8. run the frozen tuned direct ESN and shuffled control;
9. build the exact-common-row canonical comparison;
10. validate the complete classical result family;
11. run exact canonical Case151 QRC;
12. verify the Case151 frozen metric and identity oracle.

The Agent stops at the first nonzero return code.

## Optional full-smoke commands

Only after core succeeds, and only when explicitly requested:

```bash
.venv/bin/python scripts/runs/run_submission_benchmarks.py \
  all \
  --profile smoke \
  --run-id <RUN_ID> \
  --run-dir results/runs/<RUN_ID> \
  --resume \
  --reuse-existing
```

Then validate without recomputation:

```bash
.venv/bin/python scripts/runs/run_submission_benchmarks.py \
  validate-existing \
  --profile smoke \
  --run-id <RUN_ID> \
  --run-dir results/runs/<RUN_ID>
```

The primary benchmark profile is outside the default Agent workflow.

## Aggregate output

```text
results/runs/<RUN_ID>/
    run_manifest.json
    logs/
    files/
        data/
        classical_baselines/
        qrc/
            simulation/run/<RUN_ID>/
            hardware/<HARDWARE_RUN_ID>/
        quantum_studies/
            benchmark_manifest.json
            benchmark_artifact_inventory.json
            noise/run/<RUN_ID>/
            scaling/run/<RUN_ID>/
            shots/run/<RUN_ID>/
        mnist/
            raw/
            shards/
            run/<RUN_ID>/
```

Only stages included in the requested scope are created.

Data products remain under:

```text
files/data/raw/
files/data/processed/global_transition_dataset_1d/
files/data/validation/
```

Classical products remain under:

```text
files/classical_baselines/
    linear/run/<RUN_ID>/
    garch/run/<RUN_ID>/
    esn/run/<RUN_ID>/
    canonical/run/<RUN_ID>/
    validation/classical_baseline_audit.json
```

Case151 products remain under:

```text
files/qrc/simulation/run/<RUN_ID>/
```

No source code is copied into aggregate result directories.

## Classical reporting contract

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

## Core acceptance conditions

A core run is accepted only when:

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
14. `files/qrc/simulation/run/<RUN_ID>/case151_reproduction_audit.json` records `status: verified`;
15. the Case151 feature bank is `occupation_pair_raw` with width 63;
16. fold-8 Case151 selection is alpha `0.1`, lambda `0.25`;
17. Case151 records `test_rows_used: 0`;
18. no `.py` file exists beneath the aggregate run directory;
19. every command, log, and artifact belongs to the same explicit run ID.

## Full-smoke acceptance conditions

In addition to all core conditions:

1. `files/quantum_studies/benchmark_manifest.json` records `status: succeeded`, `profile: smoke`, and `validate_only: true` after the validation pass;
2. `files/quantum_studies/benchmark_artifact_inventory.json` exists;
3. every required MNIST, noise, scaling, and shot artifact exists and has a SHA-256 value;
4. every benchmark uses the same explicit run ID;
5. no hardware action occurred;
6. the primary profile was not run.

## Hardware boundary

Completed hardware evidence may be stored beneath `files/qrc/hardware/<HARDWARE_RUN_ID>/` and validated in a separate retrieval-only task. The core and full-smoke workflows must not query a backend, select a device, retrieve a job, or submit a fresh hardware job.

## Failure handling

On any failure, preserve the run directory and logs, report the first failed command and exact log path, classify the blocker, and do not patch scientific behavior during reproduction.
