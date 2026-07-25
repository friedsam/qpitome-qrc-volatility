# QPITOME QRC Volatility

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/friedsam/qpitome-qrc-volatility.git)

Phase 3 Global Industry Challenge project for qBraid / MITRE / JonesTrading, Track A: Financial Volatility Prediction.

## Scientific objective

Forecast ten-day volatility paths from ordered forty-day market histories, with primary attention to abrupt calm-to-crisis transitions at leads 1, 5, and 10. Classical HAR forecasts provide the persistence baseline; Rydberg quantum-reservoir features are evaluated as corrections to the HAR residual path.

## Judge quick start on qBraid

The **Launch on qBraid** button clones the repository into qBraid Lab when the repository is public. It does not create the project environment or install dependencies.

Run all commands from the repository root.

### 1. Create the repository-local environment

Use the standard Python virtual-environment interface. This avoids dependence on qBraid CLI environment-manager syntax, which differs between Lab images.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

If `.venv` already exists, reuse it:

```bash
source .venv/bin/activate
```

Confirm the active interpreter:

```bash
which python
python --version
python -m pip --version
```

Do not use a bare `pip` command. It may target the qBraid system interpreter rather than this repository-local environment.

### 2. Run the qBraid Skill preflight

```bash
python qbraid_skill/scripts/preflight.py --json
python -m pytest -q tests/qbraid_skill/test_skill_contract.py
```

The Agent Skills entry file is:

```text
qbraid_skill/SKILL.md
```

The current skill is intentionally Stage-1-only. It can navigate, preflight, execute, and audit the transition-data workflow. It does not yet claim to reproduce the unfinished final QRC, classical comparison, MNIST, scaling, or noise stages.

### 3. Verify the data source

The large verified transition-data fallback is not currently committed. A clean clone therefore requires working Kaggle authentication until a distributable fallback is added.

```bash
python qbraid_skill/scripts/preflight.py --strict-data-source
kaggle datasets files guillemservera/global-stock-indices-historical-data
```

Never paste, print, or commit Kaggle credentials.

### 4. Run the current Stage-1 workflow

```bash
RUN_ID="qbraid-stage1-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/runs/run_submission.py transition-data \
  --run-id "$RUN_ID" \
  --transition-source-mode live
```

Use `--transition-source-mode fallback` only when the complete verified fallback snapshot and its `fallback_manifest.json` are present. Do not use `--force` for a fresh run.

The aggregate judge-facing run is written under:

```text
results/runs/<run-id>/
```

`scripts/runs/run_submission.py` is the sole planned exception to the normal scientific-result mapping because it assembles a structured, judge-facing package. Normal scientific outputs follow:

```text
results/<domain>/<experiment>/run/<run-id>/...
```

## Current canonical Stage-1 scope

1. Acquire or restore frozen global-index OHLC inputs.
2. Apply the frozen structural-quality policy.
3. Build one-channel forty-session log-volatility sequences.
4. Construct the binary pre-control candidate pool.
5. Create eight chronologically purged walk-forward folds.
6. Rematch controls within each fold partition.
7. Validate and checksum all generated artifacts.

The former derived three-channel representation is not part of the canonical pipeline.

## Full focused validation

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

The fixed final test partition must remain unopened during development and model selection.

## Conda outside qBraid

For local development outside qBraid:

```bash
conda env create -f environment.yml
conda activate qrc-volatility
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
```

## Repository layout

```text
qbraid_skill/
src/transition_forecasting/
scripts/transition_forecasting/
tests/transition_forecasting/
results/
```

Only validated components are promoted to `stage1-dev`; only the final validated submission state is intended to be merged into `main`.
