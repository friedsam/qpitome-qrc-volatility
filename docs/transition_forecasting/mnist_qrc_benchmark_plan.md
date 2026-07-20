# MNIST QRC Benchmark Plan

## Purpose

MNIST is a standard image-classification dataset of handwritten digits. Each sample is a 28 x 28 grayscale image labeled as one of ten classes, 0 through 9.

For this project, MNIST is not the main scientific task. It is a common benchmark used to compare the expressivity of the same quantum reservoir architecture across teams and against classical baselines.

The benchmark should therefore be:

- reproducible;
- computationally bounded;
- architecturally consistent with the transition-forecasting QRC;
- easy for judges to rerun and compare;
- isolated enough that a failed MNIST experiment does not block the main forecasting results.

## Proposed benchmark pipeline

```text
MNIST image
  -> deterministic train/test subset
  -> deterministic image reduction
  -> ordered input sequence
  -> fixed Rydberg quantum reservoir
  -> measured reservoir features
  -> classical multinomial readout
  -> digit prediction
```

Only the classical readout is trained. The reservoir parameters remain fixed, as required by the QRC paradigm.

## Frozen protocol to implement

### Dataset

- Use all ten classes.
- Use the official MNIST training and test partitions.
- Select fixed, stratified subsets with a recorded random seed.
- Initial target size:
  - 2,000 training samples;
  - 1,000 test samples.
- Provide a smaller smoke configuration for development and CI:
  - 200 training samples;
  - 100 test samples.

The final subset size may be reduced if simulator runtime is excessive, but class balance and deterministic selection must be preserved.

### Image preprocessing

Each image starts as 784 pixel values. Feeding all 784 values sequentially into the reservoir would be too expensive.

Use one deterministic reduction method and freeze it before the final run. Preferred order:

1. average-pool the image to a small grid, such as 4 x 4, producing a 16-step sequence;
2. normalize pixel intensities to [0, 1];
3. flatten in row-major order to preserve spatial ordering;
4. map the sequence to the reservoir input range used by the transition task.

PCA should only be used if pooling performs poorly, because pooling is simpler, easier to explain, and does not require fitting an additional transformation.

### Reservoir

- Reuse the final persistent Rydberg reservoir implementation.
- Reuse the same physical parameterization wherever possible.
- Use the same input-injection mechanism as the main forecasting task.
- Use the same observable family and feature-extraction logic.
- Run the complete MNIST benchmark at one practical reservoir size.
- Use smaller subsets for optional reservoir-size comparisons.

The MNIST benchmark must not introduce a separate quantum architecture optimized only for images.

### Classical readout

Use multinomial logistic regression as the primary readout.

Reasons:

- natural support for ten classes;
- deterministic and fast;
- interpretable regularization;
- standard classification probabilities;
- easy comparison between raw-input and reservoir-feature baselines.

The readout regularization parameter must be selected using only the training data, ideally with a fixed internal validation split or a very small predefined grid.

### Required baselines

At minimum:

1. majority-class baseline;
2. multinomial logistic regression on the reduced image sequence without reservoir features;
3. multinomial logistic regression on Rydberg reservoir features.

An optional classical random-feature or ESN baseline may be added if already available, but it must not delay the required benchmark.

### Metrics

Report:

- test accuracy;
- macro-averaged F1 score;
- per-class precision and recall;
- confusion matrix;
- feature-generation runtime;
- readout-training runtime;
- total runtime;
- reservoir size;
- sequence length;
- number of train and test samples;
- simulator/backend configuration;
- random seed.

The primary comparison metric is test accuracy. Macro-F1 is retained to expose any class imbalance or class-specific failure.

## Runtime strategy

Quantum feature generation is the expensive stage. It must be cached separately from readout training.

Expected artifact stages:

```text
mnist_subset.npz
mnist_preprocessed.npz
mnist_reservoir_features.npz
mnist_predictions.csv
mnist_metrics.json
mnist_confusion_matrix.csv
```

If the classifier, regularization, or report code changes, reuse the cached reservoir features rather than rerunning the quantum simulation.

The feature file must include sample identifiers, labels, split assignment, reservoir configuration, and a checksum or fingerprint of the preprocessing configuration.

## Runner contract

Planned command:

```text
scripts/mnist/run_mnist_qrc_benchmark.py
```

The final submission runner should invoke it with the current run directory, for example:

```text
python scripts/mnist/run_mnist_qrc_benchmark.py --output-dir <run-dir>
```

Expected judge-facing outputs in `results/runs/<run-id>/`:

```text
mnist_metrics.json
mnist_predictions.csv
mnist_confusion_matrix.csv
mnist_reservoir_features.npz
```

The submission runner should not enable MNIST until the script exists and the smoke run passes.

## Proposed result schema

`mnist_metrics.json` should contain at least:

```json
{
  "task": "mnist_classification",
  "status": "succeeded",
  "dataset": {
    "classes": 10,
    "train_samples": 2000,
    "test_samples": 1000,
    "subset_seed": 0
  },
  "preprocessing": {
    "method": "average_pool",
    "output_shape": [4, 4],
    "sequence_length": 16
  },
  "reservoir": {
    "type": "persistent_rydberg_qrc",
    "size": null,
    "backend": null
  },
  "metrics": {
    "accuracy": null,
    "macro_f1": null
  },
  "runtime_seconds": {
    "feature_generation": null,
    "readout_training": null,
    "total": null
  }
}
```

The exact reservoir fields will be aligned with the final QRC implementation.

## Validation checks

Before accepting a final run:

- all ten classes are present in both train and test subsets;
- subset selection is deterministic;
- no test sample is used in preprocessing fitting or readout selection;
- feature rows align exactly with sample identifiers and labels;
- cached features match the current preprocessing and reservoir configuration;
- all reported metrics can be regenerated from the prediction file;
- the confusion-matrix total equals the test-sample count;
- rerunning the readout on cached features reproduces the same predictions;
- the submission runner records the command, environment, status, and required outputs.

## Execution sequence

1. Implement and run a smoke configuration.
2. Estimate feature-generation time per sample.
3. Choose the final subset size based on measured runtime.
4. Launch the full feature-generation run well before the submission deadline.
5. Inspect metrics and confusion matrix.
6. Correct protocol or implementation issues if needed.
7. Rerun only the expensive feature stage when the reservoir or preprocessing changes.
8. Integrate the successful command into the submission runner.

## Items intentionally deferred

- exact final reservoir size;
- exact simulator/backend;
- optional noise experiment;
- optional 5/10/15 reservoir-size comparison;
- optional ESN baseline;
- final subset size if runtime requires adjustment.

These depend on the final Rydberg implementation and measured throughput. The data split, preprocessing principle, readout type, required metrics, caching strategy, and runner interface should remain stable.
