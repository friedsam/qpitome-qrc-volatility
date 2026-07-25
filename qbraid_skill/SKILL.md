---
name: qpitome-qrc-volatility
description: Reproduce and audit the QPITOME quantum-reservoir volatility workflow on qBraid. Use for environment setup, repository navigation, deterministic Stage-1 execution, provenance checks, output validation, and reproducibility troubleshooting without submitting hardware jobs.
---

# QPITOME QRC Volatility Reproduction

Use this skill from the repository root. Treat the repository as a scientific submission: preserve provenance, do not silently change the experiment, and do not report success unless the recorded run manifest says the workflow succeeded and every required artifact exists.

## Current executable scope

The current `stage1-dev` implementation reproduces the transition-forecasting data stage only:

- acquisition and source verification for the global-index OHLC dataset;
- structural-quality corrections and frozen source contracts;
- the canonical one-channel 40-session volatility representation;
- binary control-candidate construction;
- eight purged walk-forward folds with fold-local control rematching;
- validation, checksums, logs, and run provenance.

The final financial QRC, classical comparison, MNIST benchmark, scaling study, and noise study are not yet wired into the automated runner on this branch. Do not claim that this skill reproduces those unfinished stages.

## Non-negotiable rules

1. Run commands from the repository root.
2. Do not modify source code, data contracts, model settings, or scientific defaults during a reproduction run.
3. Do not clean, reset, stash, or otherwise alter a dirty working tree automatically. Record it and inform the user.
4. Never select an output by taking the newest directory. Use the explicit run ID created for this execution.
5. Never infer a missing historical artifact or replace it with a regenerated file while calling it historical.
6. Do not evaluate the fixed test partition during development.
7. Do not submit a hardware job.
8. Do not place source files inside the result directory.
9. Stop on failed commands, missing required outputs, source-hash mismatches, or validation failures.

Read `references/repository-map.md` before changing or troubleshooting paths. Read `references/run-contract.md` before executing the workflow.

## Workflow

### 1. Establish repository identity

```bash
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git status --short
qbraid --version
```

Remain at the repository root returned by Git. If the working tree is dirty, continue only for inspection or after explicitly recording that the run is not from a clean checkout.

### 2. Create or activate the repository-local environment

The Launch on qBraid link clones the repository only. It does not create an environment or install dependencies.

Do not depend on `qbraid envs create`; its accepted requirement syntax varies between Lab images. Use the standard Python virtual-environment interface, which qBraid supports.

If `.venv` is absent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

If `.venv` already exists:

```bash
source .venv/bin/activate
```

Use the activated environment's Python interpreter. Do not call a bare `pip` executable.

Confirm the interpreter:

```bash
which python
python --version
python -m pip --version
```

Then run the deterministic preflight:

```bash
python qbraid_skill/scripts/preflight.py --json
```

For a full data-stage run, require a usable source:

```bash
python qbraid_skill/scripts/preflight.py --strict-data-source
```

The strict check must pass before launching the full workflow.

### 3. Verify the focused contract tests

```bash
python -m pytest -q \
  tests/qbraid_skill/test_skill_contract.py \
  tests/transition_forecasting/modeling/test_stage_d_candidate_pool.py \
  tests/transition_forecasting/modeling/test_chronological_control_matching.py \
  tests/transition_forecasting/modeling/test_chronological_rematched_dataset.py \
  tests/transition_forecasting/modeling/test_chronological_splits.py \
  tests/transition_forecasting/data/test_fold_datasets.py \
  tests/runs/test_run_submission.py
```

Do not proceed after a failed test.

### 4. Resolve the data-source mode explicitly

The large verified fallback snapshot is not currently committed to `stage1-dev`.

- If `data/fallback/transition_forecasting/global_stock_indices_historical_data/fallback_manifest.json` exists and preflight verifies it, use `fallback`.
- Otherwise, require a working Kaggle CLI and valid Kaggle authentication, verify access to `guillemservera/global-stock-indices-historical-data`, and use `live`.
- Do not use `auto` while the fallback is absent.
- Never request that credentials be pasted into chat, printed, committed, or copied into a run directory.

Verify live access without exposing secrets:

```bash
kaggle datasets files guillemservera/global-stock-indices-historical-data
```

### 5. Run the canonical Stage-1 workflow

Create one explicit UTC run ID:

```bash
RUN_ID="qbraid-stage1-$(date -u +%Y%m%dT%H%M%SZ)"
```

For the current clean-clone path, run:

```bash
python scripts/runs/run_submission.py transition-data \
  --run-id "$RUN_ID" \
  --transition-source-mode live
```

Use `--transition-source-mode fallback` only when the verified fallback is actually present. Do not add `--force` to a fresh run.

### 6. Validate the produced run

The run directory is:

```text
results/runs/<run-id>/
```

Validate all of the following:

- `run_manifest.json` exists and has `status: succeeded`;
- every entry under `required_outputs` has `exists: true` and a SHA-256 value;
- the data-pipeline audit has `passed: true`;
- the checksum report has `passed: true`;
- `test_evaluated` remains false;
- the recorded control protocol is `precontrol_binary_matching`;
- the recorded representation is one-channel `log_volatility_level`;
- the recorded fold count is eight;
- no Python source file appears anywhere under the run directory.

Useful checks:

```bash
python -m json.tool "results/runs/$RUN_ID/run_manifest.json" >/dev/null
find "results/runs/$RUN_ID" -type f -name '*.py' -print
```

The `find` command must produce no output.

### 7. Report the result precisely

Return:

- repository commit and branch;
- clean or dirty working-tree state;
- Python and qBraid CLI versions;
- exact command executed;
- explicit run ID and run directory;
- workflow status;
- command runtimes from the run manifest;
- source mode used;
- validation and checksum status;
- key dataset counts from the generated manifests;
- any warning, skipped operation, or unresolved limitation.

Separate verified facts from inference. Never summarize a failed or incomplete run as successful.

## Failure handling

When a command fails:

1. stop the workflow;
2. preserve the failed run directory and logs;
3. identify the first failing command from `run_manifest.json`;
4. inspect only the corresponding log and direct dependencies;
5. report the exact error and whether it is environmental, data/provenance-related, or a code defect;
6. do not patch scientific behavior inside the reproduction run.

Code changes belong in a separate development task and a separate commit.
