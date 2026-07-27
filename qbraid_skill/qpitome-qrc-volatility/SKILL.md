---
name: qpitome-qrc-volatility
description: "Reproduce and audit the QPITOME volatility submission on qBraid, including verified financial data, frozen classical baselines, canonical Case151 QRC, and optional Phase-3 smoke studies, without opening the reserved test partition or submitting hardware jobs."
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
<REPO_ROOT>/scripts/reproduction/run_case151_simulation.py
<REPO_ROOT>/scripts/runs/run_submission_benchmarks.py
<REPO_ROOT>/config/transition_forecasting/classical_benchmarks/frozen_submission.json
<REPO_ROOT>/config/case151/expected_metrics.json
<SKILL_ROOT>/references/repository-map.md
<SKILL_ROOT>/references/run-contract.md
<SKILL_ROOT>/scripts/bootstrap.py
<SKILL_ROOT>/scripts/preflight.py
<SKILL_ROOT>/scripts/preflight_classical.py
```

If root resolution is ambiguous or a required path is absent, stop and report a path-resolution blocker.

## Executable scopes

### Default core scope

A generic request to reproduce or audit the submission means the **core** scope. It reproduces:

1. verified global-index OHLC acquisition or fallback restoration;
2. the frozen one-channel transition dataset and eight rematched folds;
3. persistence, canonical HAR, and direct sequence-ridge forecasts;
4. zero-mean Student-t GARCH(1,1) using the `arch` backend;
5. the frozen tuned direct ESN and identically specified shuffled-order control;
6. exact-common-row classical tables for Transition, L1, L5, L10, Controls, and Pooled;
7. RMSE, log-volatility QLIKE, and Mincer-Zarnowitz intercept, slope, and R²;
8. the exact six-atom canonical Case151 QRC simulation with the 63-feature `occupation_pair_raw` readout;
9. fold-specific chronological QRC readout selection, including the frozen fold-8 identity alpha `0.1` and lambda `0.25`;
10. manifests, logs, checksums, coverage, predictions, metrics, and validation reports.

The ESN is not retuned during reproduction. Parameters come only from:

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
```

The Case151 QRC is not retuned or substituted. Its frozen oracle comes only from:

```text
config/case151/expected_metrics.json
```

### Optional full-smoke scope

Run the **full-smoke** extension only when the user explicitly asks for the complete smoke study, all Phase-3 studies, or full-smoke. It adds bounded, executable checks for:

- MNIST palindrome classification;
- density-matrix noise;
- exact/resourced atom scaling;
- finite-shot direction preservation.

The smoke profile is an integration test, not a replacement for the frozen primary study sizes.

### Hardware boundary

Hardware is not part of either core or full-smoke. Existing hardware evidence may be described or packaged only in a separate retrieval-only task. Do not submit, query, or select a hardware job during this workflow.

Aggregate artifacts are organized by judge-facing pipeline stage beneath `results/runs/<RUN_ID>/files/`: `data`, `classical_baselines`, `qrc`, `quantum_studies`, and `mnist`. Do not recreate internal source-package nesting inside the aggregate output.

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
13. Do not replace canonical Case151 with a later six-mode readout or another QRC candidate.
14. Do not run the primary Phase-3 benchmark profile unless the user explicitly requests it.

Read `<SKILL_ROOT>/references/repository-map.md` before troubleshooting paths. Read `<SKILL_ROOT>/references/run-contract.md` before scientific execution.

## Procedure

All shell commands below run with working directory `REPO_ROOT`. Replace each placeholder with the same literal value in every command; do not depend on shell variables persisting between agent actions.

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

Bootstrap creates or reuses `.venv`, installs `.[test]`, runs the data and classical preflights, and executes focused data, classical, QRC, benchmark, and hardware-safety contract tests. Do not substitute a bare `pip`, system installation, `qbraid envs create`, or shell activation. Use `.venv/bin/python` afterward.

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

### 5. Validate the data and classical stages

The aggregate directory is:

```text
results/runs/<RUN_ID>/
```

Accept the data/classical stages only when:

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

### 6. Run and verify canonical Case151 QRC

Use the folds already generated in the same accepted aggregate run:

```bash
.venv/bin/python scripts/reproduction/run_case151_simulation.py \
  --fold-dir results/runs/<RUN_ID>/files/data/processed/global_transition_dataset_1d/purged_walk_forward_folds \
  --output-root results/runs/<RUN_ID>/files/qrc/simulation/run \
  --run-id <RUN_ID>
```

The script runs the frozen exact simulation and fails closed against the Case151 oracle. The output directory is:

```text
results/runs/<RUN_ID>/files/qrc/simulation/run/<RUN_ID>/
```

Accept Case151 only when:

- `case151_reproduction_audit.json` records `status: verified`;
- `feature_bank` is `occupation_pair_raw`;
- `feature_width` is exactly `63`;
- fold-8 selected alpha is `0.1`;
- fold-8 selected lambda is `0.25`;
- `test_rows_used` is `0`;
- `pooled_metrics.csv`, `readout_selections.csv`, `feature_diagnostics.csv`, and `summary.json` exist;
- the output is beneath the same literal aggregate run ID;
- no source file appears beneath the aggregate run.

Check at minimum:

```bash
.venv/bin/python -m json.tool results/runs/<RUN_ID>/files/qrc/simulation/run/<RUN_ID>/case151_reproduction_audit.json >/dev/null
find results/runs/<RUN_ID> -type f -name '*.py' -print
```

A completed and accepted data/classical run may be resumed at this step after an Agent interruption. Revalidate Step 5 using the exact recorded run ID, then continue; do not rerun classical merely because the Agent session changed.

### 7. Optional full-smoke extension

Only for an explicit full-smoke request, run:

```bash
.venv/bin/python scripts/runs/run_submission_benchmarks.py \
  all \
  --profile smoke \
  --run-id <RUN_ID> \
  --run-dir results/runs/<RUN_ID> \
  --resume \
  --reuse-existing
```

Then perform a read-only validation pass:

```bash
.venv/bin/python scripts/runs/run_submission_benchmarks.py \
  validate-existing \
  --profile smoke \
  --run-id <RUN_ID> \
  --run-dir results/runs/<RUN_ID>
```

Accept full-smoke only when:

- `files/quantum_studies/benchmark_manifest.json` records `status: succeeded`, `profile: smoke`, and `validate_only: true` after validation;
- `files/quantum_studies/benchmark_artifact_inventory.json` exists;
- all required MNIST, noise, scaling, and shot outputs exist and have SHA-256 values;
- only the smoke profile was run;
- every benchmark stage uses the same literal run ID;
- no hardware action occurred;
- no source file appears beneath the aggregate run.

### 8. Report precisely

For core, return:

- commit, branch, and working-tree state;
- Python and qBraid versions;
- bootstrap, preflight, and focused-test results;
- requested and used source modes and fallback-substitution status;
- exact run ID and path;
- command statuses and runtimes;
- data audit, checksum, and classical audit status;
- GARCH coverage and backend;
- the canonical Transition/L1/L5/L10/Controls/Pooled classical metrics table;
- Case151 audit identity, feature width, fold-8 alpha/lambda, test-row count, and headline QLIKE/RMSE metrics;
- skipped operations, warnings, and unresolved limitations.

For full-smoke, additionally return:

- each benchmark command and runtime;
- MNIST, noise, scaling, and shot output paths;
- benchmark validation and artifact-inventory status;
- the bounded smoke profile limitations.

Never summarize a failed, blocked, or partial run as successful.

## Failure handling

On failure:

1. stop;
2. preserve the failed run directory and logs;
3. identify the first failing command;
4. classify the failure as path resolution, environment, data/provenance, classical computation, Case151 QRC, benchmark computation, or validation;
5. report the exact error and log path;
6. do not patch scientific behavior during reproduction.

Development fixes belong in a separate task and commit.
