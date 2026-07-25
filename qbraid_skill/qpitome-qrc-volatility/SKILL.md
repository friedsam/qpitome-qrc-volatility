---
name: qpitome-qrc-volatility
description: Reproduce and audit the QPITOME quantum-reservoir volatility submission on qBraid. Use when an agent must set up the repository, validate provenance and scientific identity, run the canonical workflow, and report reproducibility without modifying science or submitting hardware jobs.
---

# QPITOME QRC Volatility Reproduction

Operate from the repository root. This is an agent-executable workflow: create the environment, install dependencies, run checks, execute the canonical runner, validate outputs, and report the result. Do not ask the judge to run terminal commands for you.

## Current executable scope

The current branch reproduces the transition-forecasting data stage only:

- acquisition and source verification for global-index OHLC data;
- frozen structural-quality corrections;
- one-channel 40-session log-volatility sequences;
- binary control-candidate construction;
- eight purged walk-forward folds with fold-local rematching;
- validation, checksums, logs, and run provenance.

The final financial QRC, classical comparison, MNIST benchmark, scaling study, noise study, and final artifact collection are not yet wired into this branch. Do not claim otherwise.

## Non-negotiable rules

1. Run from the repository root.
2. Do not modify source code, contracts, parameters, data policy, or scientific defaults during reproduction.
3. Do not clean, reset, stash, or otherwise alter a dirty working tree automatically. Record it.
4. Use explicit run IDs. Never select the newest result directory.
5. Never infer a missing historical artifact or represent regenerated evidence as historical.
6. Do not evaluate the fixed test partition during development.
7. Do not submit or query a hardware job.
8. Do not place source files in result directories.
9. Stop on the first failed command, missing required output, hash mismatch, or validation failure.
10. Report verified facts separately from limitations and unresolved blockers.

Read `references/repository-map.md` before troubleshooting paths. Read `references/run-contract.md` before scientific execution.

## Procedure

### 1. Establish repository identity

Run:

```bash
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git status --short
qbraid --version
```

Remain at the Git root. A dirty tree does not authorize cleanup; record the state and continue only with non-destructive inspection unless the user explicitly approves development changes.

### 2. Bootstrap and verify the execution environment

The agent owns environment setup. Execute exactly:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json
```

This idempotent helper:

- creates `.venv` when absent;
- installs the repository and test dependencies into `.venv`;
- runs the deterministic preflight;
- runs the focused Stage-1 and skill contract tests;
- stops at the first failure.

Do not substitute `qbraid envs create`, a bare `pip`, or a system-wide installation. Do not rely on shell activation persisting between agent actions. Use `.venv/bin/python` and `.venv/bin/kaggle` explicitly afterward.

If bootstrap fails, report its first failing command and stop. Do not improvise another environment strategy inside the reproduction run.

### 3. Resolve the data source

Run:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
```

The preferred final-submission path is the committed verified fallback. Use `fallback` only when this manifest and its complete snapshot exist:

```text
data/fallback/transition_forecasting/global_stock_indices_historical_data/fallback_manifest.json
```

Otherwise, use `live` only when preflight confirms secure Kaggle credentials and this read-only access check succeeds:

```bash
.venv/bin/kaggle datasets files guillemservera/global-stock-indices-historical-data
```

Never ask for credentials to be pasted into chat, printed, committed, or copied into a run directory. If neither source is available, stop and report exit code 2 as a data/provenance blocker.

### 4. Run the canonical Stage-1 workflow

Generate one explicit UTC run ID:

```bash
date -u +qbraid-stage1-%Y%m%dT%H%M%SZ
```

Copy that literal value into the command. Do not rely on a shell variable surviving across agent actions.

For a verified fallback:

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \
  --run-id <RUN_ID> \
  --transition-source-mode fallback
```

For verified live access:

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \
  --run-id <RUN_ID> \
  --transition-source-mode live
```

Do not use `auto` while source availability is ambiguous. Do not add `--force` to a new run.

### 5. Validate the run

The run directory is:

```text
results/runs/<RUN_ID>/
```

Accept the run only if all conditions hold:

- `run_manifest.json` records `status: succeeded`;
- every required output exists and has a SHA-256 value;
- the data audit records `passed: true`;
- the checksum report records `passed: true`;
- `test_evaluated` remains false;
- the representation is one-channel `log_volatility_level`;
- the fold count is eight;
- the control protocol is `precontrol_binary_matching`;
- no `.py` file exists beneath the run directory;
- every command, log, and artifact belongs to the same explicit run ID.

Run these direct checks with the literal run ID:

```bash
.venv/bin/python -m json.tool results/runs/<RUN_ID>/run_manifest.json >/dev/null
find results/runs/<RUN_ID> -type f -name '*.py' -print
```

The `find` command must print nothing.

### 6. Report precisely

Return:

- commit, branch, and clean/dirty state;
- Python and qBraid CLI versions;
- exact commands executed;
- source mode;
- run ID and run directory;
- bootstrap, test, workflow, audit, and checksum status;
- command runtimes from the manifest;
- key dataset counts;
- warnings, skipped operations, and unresolved limitations.

Never summarize a failed, blocked, or partial run as successful.

## Failure handling

On failure:

1. stop;
2. preserve any failed run directory and logs;
3. identify the first failing command;
4. classify the failure as environment, data/provenance, or code;
5. report the exact error;
6. do not patch scientific behavior during reproduction.

Development fixes belong in a separate task and commit.
