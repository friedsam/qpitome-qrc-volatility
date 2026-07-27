# qBraid State-Free Submission Reproduction Contract

## Purpose

This contract defines one executable qBraid workflow for:

```text
verified transition data
+ frozen classical comparison
+ canonical Case151 QRC
+ bounded MNIST/noise/scaling/shot smoke studies
+ read-only artifact validation
```

The default judge-facing scope is `full-smoke`. An explicit `core` scope stops after Case151 verification. Hardware is outside both scopes.

## Agent-owned setup

The Agent creates or reuses the repository-local environment:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json
```

Bootstrap installs `.[test]`, runs both preflights, validates the Skill, and executes focused data, classical, QRC, benchmark, orchestration, and hardware-safety tests. Subsequent commands use `.venv/bin/python` explicitly.

## Data authority

The transition workflow uses:

```text
guillemservera/global-stock-indices-historical-data
```

Accepted source policy:

1. the committed fallback is usable only after manifest, file-set, hash, and schema verification;
2. `auto` compares the anonymous live candidate with the fallback and substitutes the fallback before installation when any file differs;
3. without a fallback, a live candidate must match the frozen raw inventory exactly;
4. the installed source is verified again;
5. the acquisition manifest records candidate comparison, installed-source comparison, fallback substitution, and substitution reason.

Resolve the source mode with:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
```

Use only the reported `auto`, `fallback`, or `live` value. Exit code 2 is a data/provenance blocker.

## Frozen authorities

Classical parameters:

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
```

Canonical Case151 identity and metric oracle:

```text
scripts/reproduction/run_case151_simulation.py
config/case151/expected_metrics.json
config/case151/agent_run_spec.json
```

The canonical classical set is:

```text
persistence
har
sequence_ridge
garch_1_1_t
esn_direct_tuned
esn_shuffled_tuned
```

GARCH must use `arch`. The fixed financial test partition remains unopened.

Case151 identity:

- exact six-atom simulator;
- A/4 → B/2 → A/4 palindrome;
- three probe times;
- `occupation_pair_raw` feature bank;
- 63 occupation/pair features;
- fold-specific chronological alpha/lambda selection;
- fold-8 alpha `0.1`, lambda `0.25`;
- intercept-free residual head;
- zero financial test rows.

## Single state-free command

Do not create or depend on shell variables. Do not generate a run ID in Bash. Execute:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope full-smoke \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

The orchestrator:

1. generates one literal UTC run ID internally;
2. computes every path from that ID inside one Python process;
3. calls `scripts/runs/run_submission_stage_layout.py financial-classical`;
4. validates the aggregate and classical audit;
5. calls `scripts/reproduction/run_case151_simulation.py`;
6. validates the frozen Case151 oracle;
7. calls `scripts/runs/run_submission_benchmarks.py all --profile smoke` with resume/reuse enabled;
8. calls `scripts/runs/run_submission_benchmarks.py validate-existing --profile smoke`;
9. validates the benchmark manifest and artifact inventory;
10. rejects source files beneath the aggregate result;
11. writes `agent_scope_manifest.json`;
12. stops after the first failure.

For explicitly requested core-only execution:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope core \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

To resume a known accepted `financial-classical` run, supply its exact literal ID:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope full-smoke \
  --transition-source-mode <MODE_FROM_PREFLIGHT> \
  --run-id <EXACT_RUN_ID> \
  --resume-existing
```

The resume path never discovers a run by recency.

## Aggregate output

```text
results/runs/<RUN_ID>/
    run_manifest.json
    agent_scope_manifest.json
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

Only explicitly executed stages are created. No source code is copied into aggregate result directories.

## Classical reporting contract

Canonical groups:

```text
Transition
L1
L5
L10
Controls
Pooled
```

Controls are never subdivided by lead. Metrics are RMSE, log-volatility QLIKE, and Mincer-Zarnowitz alpha, beta, and R². The canonical comparison scores the exact common finite `(fold, sample_id)` intersection across all six classical models.

## Acceptance conditions

A default full-smoke run is accepted only when:

1. `run_manifest.json` records `status: succeeded` and `workflow: financial-classical`;
2. `test_evaluated` is false in aggregate and classical audits;
3. every required data and classical output exists and has a SHA-256 value;
4. raw acquisition verifies the installed source against the authoritative reference;
5. data audit and checksum reports pass;
6. `classical_baseline_audit.json` records `passed: true`;
7. one channel, forty input sessions, ten target sessions, and eight folds are preserved;
8. predictions contain validation rows only from folds 4–8;
9. model and reporting group sets exactly match this contract;
10. no control-by-lead row exists;
11. GARCH records backend `arch`;
12. ESN parameters match the frozen specification;
13. Case151 records `status: verified`;
14. Case151 records `occupation_pair_raw`, width 63, fold-8 alpha 0.1, lambda 0.25, and zero test rows;
15. `benchmark_manifest.json` records `status: succeeded`, `profile: smoke`, and `validate_only: true` after the final pass;
16. MNIST, noise, scaling, and shot required outputs exist;
17. `benchmark_artifact_inventory.json` exists;
18. `agent_scope_manifest.json` records `scope: full-smoke`, `status: succeeded`, all command records, and `hardware_actions_performed: false`;
19. no `.py` file exists beneath the aggregate run;
20. every command and artifact uses the same literal run ID.

A core-only run applies conditions 1–14 and 19–20 and records `scope: core`.

## Hardware boundary

Hardware is never part of `core` or `full-smoke`. The orchestrator contains no hardware command. Completed evidence may be handled only through a separate retrieval-only task. No backend query, selection, retrieval, packaging, or submission occurs during Agent reproduction.

## Failure handling

On failure:

1. preserve the run directory and logs;
2. preserve `agent_scope_manifest.json` when the run directory exists;
3. report the first failed command and return code;
4. classify the blocker as path resolution, environment, data/provenance, classical computation, Case151 QRC, benchmark computation, or validation;
5. do not alter scientific behavior during reproduction.
