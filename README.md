# QPITOME QRC Volatility

Phase 3 Global Industry Challenge project for qBraid / MITRE / JonesTrading, Track A: Financial Volatility Prediction.

## Scientific objective

Forecast ten-day volatility paths from ordered forty-day market histories, with primary attention to abrupt calm-to-crisis transitions at leads 1, 5, and 10. Classical HAR forecasts provide the persistence baseline; Rydberg quantum-reservoir features are evaluated as corrections to the HAR residual path.

## Canonical branch flow

Development components are completed and validated on `stage1-dev`. Only that branch is intended to be merged into `main` for the final submission.

The canonical Stage 1 data path is deliberately small:

1. acquire or restore the frozen global-index OHLC inputs;
2. apply the frozen structural-quality policy;
3. build one-channel forty-day log-volatility sequences;
4. construct the binary pre-control candidate pool;
5. create chronologically purged walk-forward folds;
6. validate and checksum the resulting artifacts.

The former derived three-channel representation is not built by the canonical pipeline. It was redundant for the retained QRC representation and added avoidable runtime and storage cost.

## Environment

```bash
conda env create -f environment.yml
conda activate qrc-volatility
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
```

## Rebuild the transition data stage

From the repository root:

```bash
python scripts/runs/run_submission.py transition-data \
  --run-id precontrol-rebuild-001 \
  --transition-source-mode fallback \
  --force
```

Use `--transition-source-mode live` only when the external source is available and authenticated. The workflow writes acquisition logs, the one-channel processed dataset, candidate pool, folds, validation report, checksums, and a run manifest under `results/runs/<run-id>/`.

The scientific result hierarchy remains separate:

```text
results/<area>/<script-name>/<run-id>/...
```

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

Focused Stage 1 checks run through:

```bash
python -m pytest -q \
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
