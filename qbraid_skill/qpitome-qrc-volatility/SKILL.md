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
<REPO_ROOT>/config/case151/agent_run_spec.json
<REPO_ROOT>/reference/case151/freeze_001/case151_freeze.npz
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
8. current-pipeline Case151 verification of the frozen model, chronological selection procedure, fixed alpha/lambda grids, zero financial test rows, and immutable historical reference hashes;
9. explicit metric and selected-hyperparameter deltas between the current generated folds and the historical Case151 oracle, without relabelling the current run as exact historical reproduction;
10. bounded MNIST, noise, scaling, and finite-shot smoke studies;
11. a read-only `validate-existing` pass and SHA-256 artifact inventory.

The historical metric and fold-8 selection oracle in `config/case151/expected_metrics.json` remains tied to canonical commit `40ec805cc2b4efe416c0a57f1c599cca6def92c3` and source run `palindrome_real_task_002`. On that original fold lineage, fold 8 selected alpha `0.1` and lambda `0.25`. The aggregate Agent run uses newly generated current-pipeline folds, so it verifies the same frozen search grids and selection algorithm and records the selected values rather than requiring a different fold composition to choose the same grid point.

The smoke profile is an executable integration test. It does not replace the frozen primary benchmark sizes.

### Core-only scope

Run **core** only when explicitly requested. It stops after accepted data, classical baselines, canonical Case151 QRC, and current-pipeline Case151 verification.

### Hardware boundary

Hardware is not part of full-smoke or core. Do not submit, query, select, retrieve, or package hardware jobs during this workflow. Existing Aquila evidence under `files/qrc/hardware/` is a separate retrieval-only task.

## Non-negotiable rules

1. Run every command with working directory `REPO_ROOT`.
2. Resolve bundled references and scripts from `SKILL_ROOT`.
3. Do not modify source code, contracts, parameters, data policy, or scientific defaults during reproduction.
4. Do not clean, reset, stash, or otherwise alter a dirty working tree automatically. Record it.
5. Never select the newest result directory.
6. Never infer a missing historical artifact or represent regenerated evidence as historical.
7. Never apply historical Case151 metric or selected-hyperparameter assertions to a different fold lineage and describe the mismatch as a model failure.
8. Do not evaluate the fixed financial test partition.
9. Do not perform any hardware action.
10. Do not place source files in result directories.
11. Stop on the first failed command, missing output, hash mismatch, or validation failure.
12. Do not replace the `arch` GARCH backend with the development SciPy fallback.
13. Do not replace canonical Case151 with a later six-mode readout or another QRC candidate.
14. Do not run the primary Phase-3 profile unless explicitly requested.
15. Never summarize a failed, blocked, or partial run as successful.

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

Bootstrap creates or reuses `.venv`, installs `.[test]`, runs the data and classical preflights, and executes focused data, classical, QRC, benchmark, orchestrator, fold-lineage, and hardware-safety contract tests. Do not substitute a bare `pip`, system installation, `qbraid envs create`, or shell activation. Use `.venv/bin/python` afterward.

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
scripts/reproduction/run_case151_simulation.py --verification-mode current-pipeline --archive-existing-failed
scripts/runs/run_submission_benchmarks.py all --profile smoke --resume --reuse-existing
scripts/runs/run_submission_benchmarks.py validate-existing --profile smoke
```

The current-pipeline Case151 audit must record feature width `63`, the observed fold-8 alpha and lambda, confirmation that both belong to the frozen grids, and `test_rows_used: 0`. Exact fold-8 alpha `0.1` and lambda `0.25` are required only by `historical-oracle` mode on the canonical historical fold lineage.

### 5. Resume an accepted classical run after interruption or QRC failure

Resume only with an exact recorded literal run ID:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope full-smoke \
  --transition-source-mode <MODE_FROM_PREFLIGHT> \
  --run-id <EXACT_RUN_ID> \
  --resume-existing
```

The orchestrator first revalidates `financial-classical`. It skips only a current-pipeline Case151 audit already verified under the repaired contract. If an incomplete or failed Case151 output exists, the runner moves it beneath `files/qrc/simulation/run/failed_attempts/` before creating a new attempt. It reuses complete smoke artifacts and never discovers a run by recency.

## Acceptance conditions

The workflow is accepted only when:

- `run_manifest.json` records `status: succeeded`, `workflow: financial-classical`, and `test_evaluated: false`;
- data audit and checksum reports pass;
- `classical_baseline_audit.json` records `passed: true` and the frozen six-model contract;
- GARCH records backend `arch` and ESN records the frozen specification;
- Case151 records `status: verified`, `verification_mode: current-pipeline`, `feature_bank: occupation_pair_raw`, feature width `63`, `fold8_selection_grid_verified: true`, and `test_rows_used: 0`;
- the observed fold-8 alpha belongs to `[0.1, 1.0, 10.0, 100.0, 1000.0]` and the observed lambda belongs to `[0.0, 0.25, 0.5, 1.0]`;
- Case151 records the historical fold-8 reference `0.1/0.25`, whether the current selection matches it, `historical_reference_hashes_verified: true`, and `historical_metric_oracle_applied: false`;
- Case151 includes current observed metrics and explicit deltas versus the unchanged historical reference metrics;
- for full-smoke, the final benchmark manifest records `status: succeeded`, `profile: smoke`, and `validate_only: true`;
- MNIST, noise, scaling, and shot required outputs exist and have artifact inventory entries;
- `agent_scope_manifest.json` records the requested scope, all command return codes, `hardware_actions_performed: false`, `historical_case151_metric_oracle_relabelled: false`, and `status: succeeded`;
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
- canonical classical metrics;
- current-pipeline Case151 QLIKE/RMSE metrics, selected alpha/lambda, and deltas versus the historical Case151 oracle, clearly labelled as different fold lineages;
- immutable historical reference hash status;
- MNIST, noise, scaling, and shot output paths and smoke-profile limitations;
- skipped operations, warnings, and unresolved limitations.

## Failure handling

On failure:

1. stop;
2. preserve the failed run directory and logs;
3. preserve an incomplete Case151 attempt under `failed_attempts/` before a targeted retry;
4. identify the first failing command;
5. report `agent_scope_manifest.json` when present;
6. classify the blocker as path resolution, environment, data/provenance, classical computation, Case151 QRC, benchmark computation, or validation;
7. do not patch scientific behavior during reproduction.

Development fixes belong in a separate task and commit.
