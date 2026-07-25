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

From the returned absolute path, establish:

- `SKILL_ROOT`: the directory containing this `SKILL.md`;
- `REPO_ROOT`: two directory levels above `SKILL_ROOT`.

Resolve every relative skill path against `SKILL_ROOT`. For file-read operations, use the resulting absolute path. For shell commands, set the command working directory to `REPO_ROOT`. Do not rely on shell variables or `cd` state persisting between separate agent actions.

Verify these paths before proceeding:

```text
<REPO_ROOT>/pyproject.toml
<REPO_ROOT>/scripts/runs/run_submission.py
<SKILL_ROOT>/references/repository-map.md
<SKILL_ROOT>/references/run-contract.md
<SKILL_ROOT>/scripts/bootstrap.py
<SKILL_ROOT>/scripts/preflight.py
```

If the skill cannot be located uniquely or these roots cannot be established, stop and report a path-resolution blocker. Do not improvise a different layout.

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

```bash
pwd
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git status --short
qbraid --version
```

Confirm that `pwd` and `git rev-parse --show-toplevel` identify the same root. Record a dirty tree; do not clean it.

### 2. Bootstrap and verify the execution environment

The agent owns environment setup. Execute exactly:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json
```

The helper creates `.venv` when absent, installs the repository and test dependencies, runs preflight, runs the focused Stage-1 and skill tests, and stops at the first failure.

Do not substitute `qbraid envs create`, a bare `pip`, or a system-wide installation. Do not rely on shell activation persisting between agent actions. Use `.venv/bin/python` and `.venv/bin/kaggle` explicitly afterward.

If bootstrap fails, report its first failing command and stop. Do not improvise another environment strategy.

### 3. Resolve the data source

Run:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
```

Preflight performs a read-only anonymous Kaggle `datasets files` probe. Kaggle credentials are not required when the public endpoint succeeds; any configured credentials remain informational and must never be printed or requested in chat.

The verified fallback is valid only when its manifest and complete snapshot pass hash and schema verification:

```text
data/fallback/transition_forecasting/global_stock_indices_historical_data/fallback_manifest.json
```

Use the mode reported by preflight:

- `auto`: anonymous Kaggle access and a verified fallback are both available. The runner downloads a live candidate, compares it with the fallback, and installs the fallback instead if any file is missing, extra, or changed.
- `fallback`: only the verified fallback is available.
- `live`: only anonymous live access is available. The downloaded candidate must match the frozen raw inventory exactly before installation.

If no verified mode is available, stop and report exit code 2 as a data/provenance blocker.

### 4. Run the canonical Stage-1 workflow

Generate one explicit UTC run ID:

```bash
date -u +qbraid-stage1-%Y%m%dT%H%M%SZ
```

Copy the literal value into the command; do not rely on shell state.

For preflight mode `auto`:

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \
  --run-id <RUN_ID> \
  --transition-source-mode auto
```

For preflight mode `fallback` or `live`, replace the final argument with that literal mode. Do not add `--force` to a new run.

The raw acquisition manifest must show:

- `authoritative_source_verified: true`;
- `installed_source_comparison.matched: true`;
- the candidate comparison;
- whether fallback substitution occurred;
- the substitution reason when applicable.

### 5. Validate the run

The run directory is:

```text
results/runs/<RUN_ID>/
```

Accept the run only if:

- `run_manifest.json` records `status: succeeded`;
- every required output exists and has a SHA-256 value;
- the raw acquisition manifest verifies the installed source against the authoritative reference;
- the data audit records `passed: true`;
- the checksum report records `passed: true`;
- `test_evaluated` remains false;
- the representation is one-channel `log_volatility_level`;
- the fold count is eight;
- the control protocol is `precontrol_binary_matching`;
- no `.py` file exists beneath the run directory;
- every command, log, and artifact belongs to the same explicit run ID.

```bash
.venv/bin/python -m json.tool results/runs/<RUN_ID>/run_manifest.json >/dev/null
find results/runs/<RUN_ID> -type f -name '*.py' -print
```

The `find` command must print nothing.

### 6. Report precisely

Return commit, branch, tree state, Python and qBraid versions, exact commands, requested and used source modes, candidate and installed-source comparison results, fallback substitution status, run ID and path, test/workflow/audit/checksum status, runtimes, key counts, warnings, skipped operations, and unresolved limitations.

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
