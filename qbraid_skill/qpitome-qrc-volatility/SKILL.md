---
name: qpitome-qrc-volatility
description: "Reproduce and audit the QPITOME volatility submission on qBraid, including verified financial data, frozen classical baselines, canonical Case151 QRC, and bounded Phase-3 smoke studies, without opening the reserved financial test partition or performing hardware actions."
---

# QPITOME QRC Volatility Reproduction

## Path resolution — perform this first

The qBraid agent may start in `/home/jovyan` rather than in the cloned repository. Do not assume the current working directory is the repository root.

Locate this skill by absolute path:

```bash
find /home/jovyan -maxdepth 6 -type f \
  -path '*/qbraid_skill/qpitome-qrc-volatility/SKILL.md' \
  -print -quit
```

From the returned path establish:

- `SKILL_ROOT`: the directory containing `SKILL.md`;
- `REPO_ROOT`: two directory levels above `SKILL_ROOT`.

Resolve every relative skill path against `SKILL_ROOT`. For file reads, use absolute paths. For every command, set the working directory explicitly to `REPO_ROOT`. Do not rely on `cd`, shell variables, exported environment variables, or environment activation persisting between agent actions.

Verify these files exist:

```text
<REPO_ROOT>/pyproject.toml
<REPO_ROOT>/scripts/runs/run_submission_stage_layout.py
<REPO_ROOT>/scripts/reproduction/run_case151_simulation.py
<REPO_ROOT>/scripts/runs/run_submission_benchmarks.py
<REPO_ROOT>/config/transition_forecasting/classical_benchmarks/frozen_submission.json
<REPO_ROOT>/config/case151/expected_metrics.json
<SKILL_ROOT>/scripts/bootstrap.py
<SKILL_ROOT>/scripts/preflight.py
<SKILL_ROOT>/scripts/preflight_classical.py
<SKILL_ROOT>/scripts/run_submission_scope.py
<SKILL_ROOT>/references/repository-map.md
<SKILL_ROOT>/references/run-contract.md
```

If root resolution is ambiguous or a required path is absent, stop and report a path-resolution blocker.

## Reproduction scopes

### Default full-smoke scope

A generic request to reproduce and audit the submission means **full-smoke**. It runs, in one state-free orchestration process:

1. verified global-index OHLC acquisition or checksum-verified fallback restoration;
2. the frozen one-channel transition dataset and eight rematched folds;
3. persistence, canonical HAR, and direct sequence-ridge forecasts;
4. zero-mean Student-t GARCH(1,1) using the required `arch` backend;
5. the frozen tuned direct ESN and identically specified shuffled-order control;
6. exact-common-row classical tables for Transition, L1, L5, L10, Controls, and Pooled;
7. the exact six-atom canonical Case151 QRC using the 63-feature `occupation_pair_raw` readout;
8. frozen Case151 verification, including fold-8 alpha `0.1`, lambda `0.25`, and zero financial test rows;
9. bounded MNIST, noise, scaling, and finite-shot smoke studies;
10. a read-only `validate-existing` pass and SHA-256 artifact inventory.

The smoke profile is an executable integration test. It does not replace the frozen primary benchmark sizes.

### Core-only scope

Run **core** only when explicitly requested. It stops after accepted data, classical baselines, canonical Case151 QRC, and Case151 verification.

### Hardware boundary

Hardware is not part of full-smoke or core. Do not submit, query, select, retrieve, or package hardware jobs during this workflow. Existing Aquila evidence under `files/qrc/hardware/` is a separate retrieval-only task.

## Non-negotiable rules

1. Run every command with working directory `REPO_ROOT`.
2. Resolve bundled references and scripts from `SKILL_ROOT`.
3. Do not modify source code, contracts, parameters, data policy, or scientific defaults during reproduction.
4. Do not clean, reset, stash, or otherwise alter a dirty working tree automatically. Record it.
5. Never select the newest result directory.
6. Never infer a missing historical artifact or represent regenerated evidence as historical.
7. Do not evaluate the fixed financial test partition.
8. Do not perform any hardware action.
9. Do not place source files in result directories.
10. Stop on the first failed command, missing output, hash mismatch, or validation failure.
11. Do not replace the `arch` GARCH backend with the development SciPy fallback.
12. Do not replace canonical Case151 with a later six-mode readout or another QRC candidate.
13. Do not run the primary Phase-3 profile unless explicitly requested.
14. Never summarize a failed, blocked, or partial run as successful.

Read `<SKILL_ROOT>/references/repository-map.md` before troubleshooting paths. Read `<SKILL_ROOT>/references/run-contract.md` before scientific execution.

## Procedure

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

Bootstrap creates or reuses `.venv`, installs `.[test]`, runs the data and classical preflights, and executes focused data, classical, QRC, benchmark, orchestrator, and hardware-safety contract tests. Do not substitute a bare `pip`, system installation, `qbraid envs create`, or shell activation. Use `.venv/bin/python` afterward.

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

### 4. Execute the complete state-free workflow

For the default full-smoke reproduction, execute one command:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope full-smoke \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

For an explicitly requested core-only reproduction, use:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope core \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

Do not create `RUN_ID` or `RUN_DIR` shell variables. The orchestrator generates one literal run ID, computes every path internally, calls the canonical `financial-classical`, Case151, and benchmark scripts, validates each stage, stops on the first failure, and prints the exact run ID and directory.

The underlying scientific entry points remain:

```text
scripts/runs/run_submission_stage_layout.py financial-classical
scripts/reproduction/run_case151_simulation.py
scripts/runs/run_submission_benchmarks.py all --profile smoke
scripts/runs/run_submission_benchmarks.py validate-existing --profile smoke
```

### 5. Resume an accepted classical run after interruption

Resume only with an exact recorded literal run ID:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope full-smoke \
  --transition-source-mode <MODE_FROM_PREFLIGHT> \
  --run-id <EXACT_RUN_ID> \
  --resume-existing
```

The orchestrator first revalidates `financial-classical`. It skips an already verified Case151 result, reuses complete smoke artifacts, and never discovers a run by recency.

## Acceptance conditions

The workflow is accepted only when:

- `run_manifest.json` records `status: succeeded`, `workflow: financial-classical`, and `test_evaluated: false`;
- data audit and checksum reports pass;
- `classical_baseline_audit.json` records `passed: true` and the frozen six-model contract;
- GARCH records backend `arch` and ESN records the frozen specification;
- Case151 records `status: verified`, `feature_bank: occupation_pair_raw`, feature width `63`, fold-8 alpha `0.1`, fold-8 lambda `0.25`, and `test_rows_used: 0`;
- for full-smoke, the final benchmark manifest records `status: succeeded`, `profile: smoke`, and `validate_only: true`;
- MNIST, noise, scaling, and shot required outputs exist and have artifact inventory entries;
- `agent_scope_manifest.json` records the requested scope, all command return codes, `hardware_actions_performed: false`, and `status: succeeded`;
- no `.py` file exists beneath the aggregate run directory;
- every stage uses the same literal run ID.

## Reporting

Return:

- commit, branch, and working-tree state;
- Python and qBraid versions;
- bootstrap, preflight, and focused-test results;
- requested and used data-source modes and fallback-substitution status;
- exact run ID and aggregate path;
- every command, return code, and runtime from `agent_scope_manifest.json`;
- data, checksum, classical, Case151, and benchmark audit status;
- canonical classical metrics and Case151 QLIKE/RMSE results;
- MNIST, noise, scaling, and shot output paths and smoke-profile limitations;
- skipped operations, warnings, and unresolved limitations.

## Failure handling

On failure:

1. stop;
2. preserve the failed run directory and logs;
3. identify the first failing command;
4. report `agent_scope_manifest.json` when present;
5. classify the blocker as path resolution, environment, data/provenance, classical computation, Case151 QRC, benchmark computation, or validation;
6. do not patch scientific behavior during reproduction.

Development fixes belong in a separate task and commit.
