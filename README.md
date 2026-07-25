# QPITOME QRC Volatility

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/friedsam/qpitome-qrc-volatility.git)

Phase 3 Global Industry Challenge project for qBraid / MITRE / JonesTrading, Track A: Financial Volatility Prediction.

## Scientific objective

Forecast ten-day volatility paths from ordered forty-day market histories, with primary attention to abrupt calm-to-crisis transitions at leads 1, 5, and 10. Classical HAR forecasts provide the persistence baseline; Rydberg quantum-reservoir features are evaluated as corrections to the HAR residual path.

## Canonical branch flow

Components are promoted to `stage1-dev` only after they are complete and validated. Only `stage1-dev` is intended to be merged into `main` for the final submission.

The canonical Stage 1 data path is deliberately small:

1. acquire or restore the frozen global-index OHLC inputs;
2. apply the frozen structural-quality policy;
3. build one-channel forty-day log-volatility sequences;
4. construct the binary pre-control candidate pool;
5. create chronologically purged walk-forward folds;
6. validate and checksum the resulting artifacts.

The former derived three-channel representation is not built by the canonical pipeline. It was redundant for the retained QRC representation and added avoidable runtime and storage cost.

## Environment

### qBraid Lab

The **Launch on qBraid** button clones this public repository into qBraid Lab. It does **not** create, activate, or populate the required Python environment.

From the cloned repository root, inspect the available environments:

```bash
qbraid envs list
```

If `qrc-volatility` does not exist, create it from the committed environment specification:

```bash
qbraid envs create -f environment.yml -y
```

Activate the environment and install the repository with its test dependencies:

```bash
qbraid envs activate qrc-volatility
python -m pip install -e ".[test]"
```

Use `python -m pip`, not a bare `pip` command. In qBraid Lab, a bare `pip` may target the nonpersistent system interpreter rather than the activated qBraid environment.

Verify that the correct interpreter and repository contract are active:

```bash
which python
python --version
python qbraid_skill/scripts/preflight.py --json
```

### Conda outside qBraid

Create the same environment with:

```bash
conda env create -f environment.yml
conda activate qrc-volatility
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
```

## qBraid Skill

The agent-executable skill package is rooted at:

```text
qbraid_skill/
```

Its required Agent Skills entry file is:

```text
qbraid_skill/SKILL.md
```

The current skill is intentionally Stage-1-only. It can navigate, create or activate the project environment, preflight, execute, and audit the transition-data workflow. It does not yet claim to reproduce the unfinished final QRC, classical comparison, MNIST, scaling, or noise stages.

From the repository root, test the environment and repository contract with:

```bash
python qbraid_skill/scripts/preflight.py --json
python qbraid_skill/scripts/preflight.py --strict-data-source
```

In qBraid Lab, use `qbraid skills --help` to confirm the exact local install/test syntax supported by the installed CLI version, then point it at the `qbraid_skill/` package directory.

## Rebuild the transition-data stage

The large verified transition-data fallback is not currently committed to `stage1-dev`. A clean clone therefore requires working Kaggle authentication unless the fallback is supplied separately.

Verify access first:

```bash
kaggle datasets files guillemservera/global-stock-indices-historical-data
```

Then run from the repository root with a new immutable run ID:

```bash
RUN_ID="qbraid-stage1-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/runs/run_submission.py transition-data \
  --run-id "$RUN_ID" \
  --transition-source-mode live
```

Use `--transition-source-mode fallback` only when the complete verified fallback snapshot and its `fallback_manifest.json` are present. Do not use `--force` for a fresh run.

The workflow writes acquisition logs, the one-channel processed dataset, candidate pool, folds, validation report, checksums, and a run manifest under `results/runs/<run-id>/`.

The scientific result hierarchy remains separate:

```text
results/<domain>/<experiment>/run/<run-id>/...
```

`scripts/runs/run_submission.py` is the sole planned exception because it assembles one structured, judge-facing aggregate run.

## Canonical transition-data layout

```text
src/transition_forecasting/data/
src/transition_forecasting/modeling/
scripts/transition_forecasting/data/
tests/transition_forecasting/data/
tests/transition_forecasting/modeling/
results/runs/<run-id>/
```

## Validation

Focused Stage 1 and qBraid Skill checks run through:

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

The common final test partition must remain unopened during development and model selection.

## Current recovery state

`stage1-dev` has been restored to the pre-control binary matching protocol. The abandoned calm/hard-negative redesign no longer feeds candidate construction, fold assignment, HAR fitting, or residual generation. The next gate is to rebuild the global one-channel matrix and reproduce the archived pre-control HAR and QRC results before porting further QRC work.
