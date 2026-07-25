---
name: qpitome-qrc-volatility
description: Reproduce and audit the QPITOME quantum-reservoir volatility submission on qBraid. Use when an agent must set up the repository, validate provenance and scientific identity, run the canonical workflow, and report reproducibility without modifying science or submitting hardware jobs.
---

# QPITOME QRC Volatility Reproduction

## Path resolution — perform this first

The qBraid agent may start in `/home/jovyan` rather than in the cloned repository. Do not assume the current working directory is the repository root, and do not resolve skill resources relative to the repository root.

Locate this skill by absolute path:

```bash
find /home/jovyan -maxdepth 6 -type f \
  -path '*/qbraid_skill/qpitome-qrc-volatility/SKILL.md' \
  -print -quit
```

From the returned absolute path, establish these two roots conceptually:

- `SKILL_ROOT`: the directory containing this `SKILL.md`;
- `REPO_ROOT`: two directory levels above `SKILL_ROOT`.

Resolve every relative skill path against `SKILL_ROOT`. For file-read operations, use the resulting absolute path. For shell commands, set the command working directory to `REPO_ROOT`. Do not rely on shell variables or `cd` state persisting between separate agent actions.

Before proceeding, verify that all of these exist:

```text
<REPO_ROOT>/pyproject.toml
<REPO_ROOT>/scripts/runs/run_submission.py
<SKILL_ROOT>/references/repository-map.md
<SKILL_ROOT>/references/run-contract.md
<SKILL_ROOT>/scripts/bootstrap.py
<SKILL_ROOT>/scripts/preflight.py
```

If the skill cannot be located uniquely or these roots cannot be established, stop and report a path-resolution blocker. Do not search for similarly named replacement files or improvise a different repository layout.

Operate from `REPO_ROOT`. This is an agent-executable workflow: create the environment, install dependencies, run checks, execute the canonical runner, validate outputs, and report the result. Do not ask the judge to run terminal commands for you.

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

1. Run from `REPO_ROOT`.
2. Resolve bundled references and scripts from `SKILL_ROOT`, never from `REPO_ROOT`.
3. Do not modify source code, contracts, parameters, data policy, or scientific defaults during reproduction.
4. Do not clean, reset, stash, or otherwise alter a dirty working tree automatically. Record it.
5. Use explicit run IDs. Never select the newest result directory.
6. Never infer a missing historical artifact or represent regenerated evidence as historical.
7. Do not evaluate the fixed test partition during development.
8. Do not submit or query a hardware job.
9. Do not place source files in result directories.
10. Stop on the first failed command, missing required output, hash mismatch, or validation failure.
11. Report verified facts separately from limitations and unresolved blockers.

Read `<SKILL_ROOT>/references/repository-map.md` before troubleshooting paths. Read `<SKILL_ROOT>/references/run-contract.md` before scientific execution. Replace `<SKILL_ROOT>` with the actual absolute directory in file-tool calls.

## Procedure

All shell commands below must execute with working directory `REPO_ROOT`.

### 1. Establish repository identity

Run:

```bash
pwd
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git status --short
qbraid --version
```

Confirm that `pwd` and `git rev-parse --show-toplevel` identify the same repository root. A dirty tree does not authorize cleanup; record the state and continue only with non-destructive inspection unless the user explicitly approves development changes.

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
4. classify the failure as path resolution, environment, data/provenance, or code;
5. report the exact error;
6. do not patch scientific behavior during reproduction.

Development fixes belong in a separate task and commit.
