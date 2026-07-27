---
name: qpitome-qrc-volatility
description: Reproduce and audit the QPITOME volatility submission data and frozen classical comparison on qBraid without modifying science or opening the reserved test partition.
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

From the returned path establish:

- `SKILL_ROOT`: the directory containing `SKILL.md`;
- `REPO_ROOT`: two directory levels above `SKILL_ROOT`.

Resolve every relative skill path against `SKILL_ROOT`. For file reads, use absolute paths. For shell commands, set the working directory explicitly to `REPO_ROOT`. Do not rely on `cd`, shell variables, or environment activation persisting between agent actions.

Verify:

```text
<REPO_ROOT>/pyproject.toml
<REPO_ROOT>/scripts/runs/run_submission.py
<REPO_ROOT>/scripts/runs/run_submission_layout.py
<REPO_ROOT>/scripts/runs/run_submission_stage_layout.py
<REPO_ROOT>/config/transition_forecasting/classical_benchmarks/frozen_submission.json
<SKILL_ROOT>/references/repository-map.md
<SKILL_ROOT>/references/run-contract.md
<SKILL_ROOT>/scripts/bootstrap.py
<SKILL_ROOT>/scripts/preflight.py
<SKILL_ROOT>/scripts/preflight_classical.py
```

If root resolution is ambiguous or a required path is absent, stop and report a path-resolution blocker.

## Executable scope

The canonical agent workflow now reproduces:

1. verified global-index OHLC acquisition or fallback restoration;
2. the frozen one-channel transition dataset and eight rematched folds;
3. persistence, canonical HAR, and direct sequence-ridge forecasts;
4. zero-mean Student-t GARCH(1,1) using the `arch` backend;
5. the frozen tuned direct ESN and identically specified shuffled-order control;
6. exact-common-row classical tables for Transition, L1, L5, L10, Controls, and Pooled;
7. RMSE, log-volatility QLIKE, and Mincer-Zarnowitz intercept, slope, and R²;
8. manifests, logs, checksums, coverage, predictions, and validation reports.

The ESN is not retuned during reproduction. Parameters come only from:

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
```

Financial QRC, MNIST, qubit scaling, noise studies, and fresh hardware execution are not part of this workflow yet. Do not claim otherwise.

Aggregate artifacts are organized by judge-facing pipeline stage beneath `results/runs/<RUN_ID>/files/`: `data`, `classical_baselines`, `qrc`, `quantum_studies`, and `mnist`. Do not recreate internal source-package nesting inside the aggregate output.

Committed hardware results may later be verified, summarized, and compared beneath `files/qrc/hardware/`. Never query a backend or submit a fresh hardware job during the default reproduction workflow.

## Non-negotiable rules

1. Run commands from `REPO_ROOT`.
2. Resolve bundled references and scripts from `SKILL_ROOT`.
3. Do not modify source code, contracts, parameters, data policy, or scientific defaults during reproduction.
4. Do not clean, reset, stash, or otherwise alter a dirty working tree automatically. Record it.
5. Use one explicit run ID. Never select the newest result directory.
6. Never infer a missing historical artifact or represent regenerated evidence as historical.
7. Do not evaluate the fixed test partition.
8. Do not submit, query, or select a hardware job.
9. Do not place source files in result directories.
10. Stop on the first failed command, missing required output, hash mismatch, or validation failure.
11. Report verified facts separately from limitations and unresolved blockers.
12. Do not replace the required `arch` GARCH backend with the development SciPy fallback.

Read `<SKILL_ROOT>/references/repository-map.md` before troubleshooting paths. Read `<SKILL_ROOT>/references/run-contract.md` before scientific execution.

## Procedure

All shell commands below run with working directory `REPO_ROOT`.

### 1. Record repository identity

```bash
pwd
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git status --short
qbraid --version
```

Confirm that `pwd` and `git rev-parse --show-toplevel` identify the same root. Record a dirty tree; do not clean it.

### 2. Bootstrap and verify the environment

Execute exactly:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json
```

Bootstrap creates or reuses `.venv`, installs `.[test]`, runs the data preflight, runs the classical preflight, and executes the focused contract suite. Do not substitute a bare `pip`, system installation, `qbraid envs create`, or shell activation. Use `.venv/bin/python` afterward.

If bootstrap fails, stop and report its first failed command.

### 3. Resolve the verified data source

Run:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
```

Use exactly the mode reported by preflight:

- `auto`: anonymous live access and the verified fallback are available;
- `fallback`: only the verified fallback is available;
- `live`: only anonymous live access is available.

If no verified mode is available, stop with the data/provenance blocker. Do not improvise another dataset.

### 4. Run the canonical data-plus-classical workflow

Generate one literal UTC run ID:

```bash
date -u +qbraid-financial-classical-%Y%m%dT%H%M%SZ
```

Copy the literal value into:

```bash
.venv/bin/python scripts/runs/run_submission_stage_layout.py financial-classical \
  --run-id <RUN_ID> \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

Do not add `--force` to a new run. The workflow builds the data first, then runs linear/HAR, GARCH, ESN, the canonical comparison, and the classical validator. It stops after the first nonzero command.

### 5. Validate the aggregate run

The aggregate directory is:

```text
results/runs/<RUN_ID>/
```

Accept the run only when:

- `run_manifest.json` records `status: succeeded` and `workflow: financial-classical`;
- every required output exists with a SHA-256 value;
- the acquisition manifest records authoritative-source verification and a matched installed source;
- the data audit and checksum report pass;
- `classical_baseline_audit.json` records `passed: true`;
- the canonical prediction file contains exactly the six agreed classical models;
- reporting groups are exactly Transition, L1, L5, L10, Controls, and Pooled;
- no control-by-lead row exists;
- folds are 4–8 and all prediction rows have `fold_split=val`;
- the GARCH configuration records backend `arch`;
- the ESN configuration matches the frozen specification;
- `test_evaluated` remains false;
- no `.py` file exists beneath the aggregate result directory;
- every command, log, and artifact belongs to the same literal run ID.

Check at minimum:

```bash
.venv/bin/python -m json.tool results/runs/<RUN_ID>/run_manifest.json >/dev/null
find results/runs/<RUN_ID> -type f -name '*.py' -print
```

The `find` command must print nothing.

### 6. Report precisely

Return:

- commit, branch, and working-tree state;
- Python and qBraid versions;
- bootstrap, preflight, and focused-test results;
- requested and used source modes and fallback-substitution status;
- exact run ID and path;
- command statuses and runtimes;
- data audit, checksum, and classical audit status;
- GARCH coverage and backend;
- the canonical Transition/L1/L5/L10/Controls/Pooled metrics table;
- skipped operations, warnings, and unresolved limitations.

Never summarize a failed, blocked, or partial run as successful.

## Failure handling

On failure:

1. stop;
2. preserve the failed run directory and logs;
3. identify the first failing command;
4. classify the failure as path resolution, environment, data/provenance, classical computation, or validation;
5. report the exact error and log path;
6. do not patch scientific behavior during reproduction.

Development fixes belong in a separate task and commit.
