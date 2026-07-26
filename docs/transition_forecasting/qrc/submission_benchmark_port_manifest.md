# Submission benchmark selective-port manifest

## Purpose

This file is the living, human-readable allow-list for selectively porting the final QRC benchmark work from `qrc-freeze-comparison-work` into the clean submission integration branch. The development branch must not be merged wholesale.

The repository-wide migration ledger remains authoritative for path moves. This document instead records scientific and executable dependencies for the final benchmark runner.

## Primary MNIST benchmark

### New files produced by this work

| Status | Path | Role |
|---|---|---|
| port | `src/transition_forecasting/qrc/mnist_palindrome_benchmark.py` | Deterministic two-channel MNIST adapter, resumable/sharded palindrome feature generation, strict shard merge, and final multinomial readouts. |
| port | `scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py` | Thin `shard`/`merge` command-line wrapper with the frozen six-atom A/B/A configuration. |
| port | `tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py` | Pooling, contrast, deterministic sampling, sharding, and frozen-contract tests. |
| port/update | `docs/transition_forecasting/qrc/submission_benchmark_port_manifest.md` | This selective-port inventory. |

### Existing source dependencies

| Status | Path | Role |
|---|---|---|
| port/reconcile | `src/transition_forecasting/data/mnist_acquisition.py` | IDX discovery, validation, acquisition, and loading. Replace or verify the unconfirmed Kaggle dataset identifier before final submission. |
| port | `scripts/transition_forecasting/data/acquire_mnist.py` | Thin acquisition wrapper used by the agent workflow. |
| port | `tests/transition_forecasting/data/test_mnist_acquisition.py` | IDX parsing and acquisition contract tests. |
| port | `src/transition_forecasting/qrc/palindrome_real_task_relevance_assay.py` | Canonical A/B/A schedule resolution and exact palindrome probability evolution. Consider extracting the reusable simulator from the assay module during clean-port work, without changing numerical semantics. |
| port | `src/transition_forecasting/qrc/bivariate_crossover_assay.py` | A/B/A schedule, mirrored branch drive, and `occupation_pair_raw` feature bank. |
| port | `src/transition_forecasting/qrc/bivariate_capacity_dynamics.py` | Segment evolution used by the palindrome simulator. |
| port | `src/transition_forecasting/qrc/temporal_rydberg_chain.py` | Frozen reservoir configuration and state/probe helpers. |
| port | `src/transition_forecasting/qrc/temporal_rydberg_ladder.py` | Six-atom staggered asymmetric ladder geometry and precomputation. |
| port | `src/transition_forecasting/qrc/representation_candidates.py` | Train-only robust two-channel scaler. |
| port | `src/transition_forecasting/qrc/ladder_finite_shot_sampling.py` | Probability validation used by the palindrome simulator. |

### Frozen primary protocol

- Official MNIST train and test partitions remain separate.
- Deterministic class-balanced subsets: default 2,000 train and 1,000 test.
- Average-pool 28×28 images to 4×4.
- Channel A: pooled intensity.
- Channel B: pooled cell minus the mean of its valid orthogonal neighbours.
- Flatten both channels in row-major order to a 16-step two-channel sequence.
- Fit the existing robust channel scaler on training rows only.
- Use the six-atom staggered ladder, interaction scale 1.25, 0.02 μs per input step, probes at one quarter, one half, and endpoint, and A/4→B/2→A/4 palindrome.
- Use the full hardware-natural `occupation_pair_raw` bank, not the historical six-mode compression.
- Train only the multinomial logistic readout; reservoir parameters stay fixed.
- Required baselines: majority class and logistic regression on the identical flattened two-channel input.
- Required outputs: merged features, model comparison, predictions, per-class precision/recall/F1, and one confusion matrix per model.

### Sharded execution contract

Each qBraid feature shard is an independent human-readable run directory:

```text
results/transition_forecasting/qrc/mnist_palindrome_benchmark/
    mnist_palindrome_primary_001_shard_000_of_008/
        params.json
        sample_contract.npz
        batches/
        features.npz
        summary.json
```

The merge command accepts all shard directories and writes:

```text
results/transition_forecasting/qrc/mnist_palindrome_benchmark/
    mnist_palindrome_primary_001/
        params.json
        mnist_palindrome_features.npz
        model_comparison.csv
        mnist_predictions.csv
        per_class_metrics.csv
        confusion_matrix_*.csv
        summary.json
```

The merge must reject missing shard indices, duplicate/missing global rows, mismatched dataset fingerprints, mismatched model fingerprints, inconsistent fields, non-finite features, or duplicate sample identifiers.

## Optional position-encoded MNIST benchmark

The existing position-encoded benchmark remains a separate optional result. It must not replace or be blended into the primary palindrome benchmark.

Potential selective-port files, only after the primary run is secured:

- `src/transition_forecasting/qrc/mnist_position_benchmark.py`
- `src/transition_forecasting/qrc/position_encoded_rydberg.py`
- `scripts/transition_forecasting/qrc/run_mnist_position_benchmark.py`
- `tests/transition_forecasting/qrc/test_mnist_position_benchmark.py`

## Submission-agent integration still required

After the primary MNIST smoke and full runs pass, the clean integration task must:

1. add a `mnist-palindrome` workflow to `scripts/runs/run_submission.py`;
2. add required-output validation and SHA-256 inventory for the merged result;
3. update the qBraid Skill repository map, run contract, preflight, and focused tests;
4. acquire or verify MNIST through the same deterministic data-source policy;
5. ensure the agent invokes one literal run ID and never selects the newest directory;
6. keep the optional position-encoded benchmark outside the required primary workflow.

## Not yet included

Noise, qubit scaling, and finite-shot studies will be added to this manifest only after their palindrome-specific implementations are complete.
