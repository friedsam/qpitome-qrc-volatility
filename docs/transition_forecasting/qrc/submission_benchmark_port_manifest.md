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
| port | `tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py` | Pooling, contrast, deterministic sampling, sharding, frozen-contract, and synthetic end-to-end merge tests. |
| port/update | `docs/transition_forecasting/qrc/submission_benchmark_port_manifest.md` | This selective-port inventory. |

### Existing source dependencies

| Status | Path | Role |
|---|---|---|
| port | `src/transition_forecasting/data/mnist_acquisition.py` | Anonymous Keras-hosted `mnist.npz` acquisition with the official Keras SHA-256 pin, source validation, backward-compatible IDX loading, and fallback support. |
| port | `scripts/transition_forecasting/data/acquire_mnist.py` | Thin anonymous-live/fallback acquisition wrapper used by the agent workflow. |
| port | `tests/transition_forecasting/data/test_mnist_acquisition.py` | IDX discovery plus NPZ mapping and checksum-enforcement tests. |
| port | `src/transition_forecasting/qrc/palindrome_real_task_relevance_assay.py` | Canonical A/B/A schedule resolution and exact palindrome probability evolution. Consider extracting the reusable simulator from the assay module during clean-port work, without changing numerical semantics. |
| port | `src/transition_forecasting/qrc/bivariate_crossover_assay.py` | A/B/A schedule, mirrored branch drive, and `occupation_pair_raw` feature bank. |
| port | `src/transition_forecasting/qrc/bivariate_capacity_dynamics.py` | Segment evolution used by the palindrome simulator. |
| port | `src/transition_forecasting/qrc/temporal_rydberg_chain.py` | Frozen reservoir configuration and state/probe helpers. |
| port | `src/transition_forecasting/qrc/temporal_rydberg_ladder.py` | Six-atom staggered asymmetric ladder geometry and precomputation. |
| port | `src/transition_forecasting/qrc/representation_candidates.py` | Train-only robust two-channel scaler. |
| port | `src/transition_forecasting/qrc/ladder_finite_shot_sampling.py` | Probability validation used by the palindrome simulator. |

### Frozen data source

Primary live source:

```text
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

Required SHA-256:

```text
731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1
```

The code downloads this archive anonymously with the Python standard library, checks the digest before installation, validates the canonical 60,000/10,000 arrays, and records the source URL and hash. It does not require TensorFlow, Keras, Kaggle credentials, or a committed fallback. Existing canonical IDX snapshots remain loadable as a fallback format.

Acquisition command:

```bash
python scripts/transition_forecasting/data/acquire_mnist.py \
  --destination data/raw/mnist \
  --source-mode live
```

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
- Required outputs: merged features, model comparison, predictions, per-class precision/recall/F1, shard runtime manifest, and one confusion matrix per model.

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
        shard_manifest.csv
        model_comparison.csv
        mnist_predictions.csv
        per_class_metrics.csv
        confusion_matrix_*.csv
        summary.json
```

The merge must reject missing shard indices, duplicate/missing global rows, mismatched dataset fingerprints, mismatched model fingerprints, inconsistent fields, non-finite features, or duplicate sample identifiers.

### Recommended first smoke

Run one small shard before launching the full matrix:

```bash
python scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py shard \
  --raw-dir data/raw/mnist \
  --run-id mnist_palindrome_smoke_001_shard_000_of_001 \
  --shard-index 0 \
  --shard-count 1 \
  --train-size 100 \
  --test-size 20 \
  --batch-size 20
```

Then merge it with the same explicit configuration:

```bash
python scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py merge \
  --run-id mnist_palindrome_smoke_001 \
  --shard-dirs \
    results/transition_forecasting/qrc/mnist_palindrome_benchmark/mnist_palindrome_smoke_001_shard_000_of_001 \
  --train-size 100 \
  --test-size 20 \
  --batch-size 20
```

Use the measured shard `wall_seconds / rows` to choose the final shard count. Do not enable `--include-interaction-off` in the required primary run unless capacity remains after the interaction-on result is secured.

## Optional position-encoded MNIST benchmark

The existing position-encoded benchmark remains a separate optional result. It must not replace or be blended into the primary palindrome benchmark.

Potential selective-port files, only after the primary run is secured:

- `src/transition_forecasting/qrc/mnist_position_benchmark.py`
- `src/transition_forecasting/qrc/position_encoded_rydberg.py`
- `scripts/transition_forecasting/qrc/run_mnist_position_benchmark.py`
- `tests/transition_forecasting/qrc/test_mnist_position_benchmark.py`

## Frozen palindrome noise benchmark

### New files produced by this work

| Status | Path | Role |
|---|---|---|
| port | `src/transition_forecasting/qrc/palindrome_rydberg_noise.py` | Tensorized six-atom density-matrix propagation for the exact A/4→B/2→A/4 palindrome with local T1, T2, and depolarizing channels applied after every numerical substep. |
| port | `src/transition_forecasting/qrc/palindrome_noise_assay.py` | Bounded fold-5/lead-5 robustness assay with frozen clean six-mode no-intercept Ridge readout and scenario-relative feature/forecast diagnostics. |
| port | `scripts/transition_forecasting/qrc/run_palindrome_noise_assay.py` | Thin runner with the frozen final architecture and optional exact scenario filtering for smoke runs. |
| port | `tests/transition_forecasting/qrc/test_palindrome_rydberg_noise.py` | Rotation-channel invariants, ideal statevector/density parity, noisy probability normalization, scenario selection, and architecture-drift tests. |
| port | `src/transition_forecasting/qrc/temporal_rydberg_noise.py` | Reused validated local Kraus channels, physical-time conversion, and density stabilization. |
| port | `tests/transition_forecasting/qrc/test_temporal_rydberg_noise.py` | Existing trace-preservation and channel-level tests retained as dependency coverage. |

### Frozen noise protocol

- Development-only fold 5, lead 5; no test rows.
- Balanced panel: 12 rows per class in train and validation when available.
- `level_instability` input representation with train-only robust scaling.
- Six-atom staggered ladder, interaction scale 1.25, A/4→B/2→A/4 schedule, 0.02 μs per observation, and probes at 0.25, 0.5, and 1.0.
- `six_mode_density_curvature` feature bank.
- Ridge alpha 100, `fit_intercept=False`, correction lambda 1.0.
- Fit feature scaler and readout once on ideal statevector training features; freeze both for every density/noise scenario.
- Require ideal density features to match the statevector reference before accepting noisy outputs.
- Scenarios: ideal; T1 = 200, 100, 50 μs; T2 = 100, 50 μs; total depolarizing probability = 0.005, 0.01, 0.03; and combined T1=100 μs, T2=50 μs, p=0.01.
- Interpret deltas relative to the frozen ideal QRC only. Do not infer calibrated Aquila performance or beneficial noise from this small panel.

### Verified primary result

Run ID:

```text
palindrome_noise_primary_001
```

Acceptance facts:

- 48 total development rows: 24 train and 24 validation; 14 causal residual-training rows.
- No test rows used.
- Statevector/density maximum feature mismatch: approximately 2.7e-15.
- Maximum tested relative feature MAE: approximately 0.00525 at depolarizing p=0.03.
- Minimum tested feature correlation: approximately 0.999978.
- No positive QLIKE delta versus the ideal density reference was observed; all changes were small and must be described as robustness, not noise benefit.

Expected result directory:

```text
results/transition_forecasting/qrc/palindrome_noise_assay/palindrome_noise_primary_001/
    params.json
    summary.json
    noise_metrics.csv
    predictions.csv.gz
    retained_samples.csv
    frozen_readout.npz
    features/
    plots/
```

## Submission-agent integration still required

The clean integration task must:

1. add explicit `mnist-palindrome` and `palindrome-noise` workflows to `scripts/runs/run_submission.py`;
2. add required-output validation and SHA-256 inventories for both result families;
3. update the qBraid Skill repository map, run contract, preflight, and focused tests;
4. acquire or verify MNIST through the same deterministic data-source policy;
5. ensure the agent invokes literal run IDs and never selects the newest directory;
6. keep the optional position-encoded benchmark outside the required primary workflow;
7. preserve the noise assay's development-only and frozen-readout boundaries.

## Not yet included

Qubit scaling and finite-shot studies will be added to this manifest only after their palindrome-specific implementations are complete.
