# QRC Usage Procedure

Purpose: define a clean, testable path for running the Phase 2 digital QRC logic without relying on notebook execution state.

This document is a procedural contract. Tests should protect this path first. Notebooks may remain as exploratory records, but they should not be the canonical interface for Phase 3 work.

## 1. Canonical QRC Interface

Use the module implementation, not notebooks, as the primary interface.

Current canonical module:

```text
src/qpitome_qrc/qrc/feedback_tfim_reservoir.py
```

Primary public objects/functions:

```python
from qpitome_qrc.qrc.feedback_tfim_reservoir import (
    FeedbackTFIMQRCConfig,
    build_feedback_qrc_feature_matrix,
    fit_feedback_tfim_qrc_regressor,
    run_feedback_tfim_reservoir_for_window,
    select_temporal_indices,
    summarize_feedback_qrc_result,
)
```

## 2. Minimal Single-Window Procedure

Use this for fast deterministic validation.

```python
import numpy as np

from qpitome_qrc.qrc.feedback_tfim_reservoir import (
    FeedbackTFIMQRCConfig,
    run_feedback_tfim_reservoir_for_window,
)

config = FeedbackTFIMQRCConfig(
    qubits=3,
    pca_components=2,
    lookback_days=4,
    temporal_steps=2,
    input_qubits=(0,),
    memory_qubits=(1,),
    readout_qubits=(2,),
    observable_mode="zxzz",
    feature_collection="trajectory",
    trotter_steps_per_time=1,
    evolution_time=0.10,
    disorder_strength=0.0,
    seed=42,
)

window = np.array(
    [
        [0.0, 0.1],
        [0.2, -0.1],
        [0.1, 0.0],
        [-0.2, 0.3],
    ],
    dtype=float,
)

features = run_feedback_tfim_reservoir_for_window(window, config)
```

Expected behavior:

- `features` is one-dimensional;
- all feature values are finite;
- repeated runs with the same config and input produce identical values;
- feature length is stable for the same config.

## 3. Minimal Feature-Matrix Procedure

Use this to verify batch behavior.

```python
import numpy as np

from qpitome_qrc.qrc.feedback_tfim_reservoir import (
    FeedbackTFIMQRCConfig,
    build_feedback_qrc_feature_matrix,
)

config = FeedbackTFIMQRCConfig(
    qubits=3,
    pca_components=2,
    lookback_days=4,
    temporal_steps=2,
    input_qubits=(0,),
    memory_qubits=(1,),
    readout_qubits=(2,),
    observable_mode="zxzz",
    feature_collection="trajectory",
    trotter_steps_per_time=1,
    evolution_time=0.10,
    disorder_strength=0.0,
    seed=42,
)

X_windows = np.stack([window, window + 0.05], axis=0)
H = build_feedback_qrc_feature_matrix(X_windows, config)
```

Expected behavior:

- `H` has one row per input window;
- all entries are finite;
- row width matches the single-window feature length;
- repeated runs are deterministic for fixed config and input.

## 4. Minimal Regressor Procedure

Use this after single-window and feature-matrix tests pass.

Input contract:

```python
sequence_splits = {
    "train": (X_train, y_train, train_dates),
    "val": (X_val, y_val, val_dates),
    "test": (X_test, y_test, test_dates),
}
```

Where:

- each `X_*` has shape `(samples, lookback_days, pca_components)`;
- each `y_*` has shape `(samples,)`;
- all targets are finite and positive;
- date arrays/series are chronological;
- preprocessing has already been fit on the training split only.

Call:

```python
result = fit_feedback_tfim_qrc_regressor(
    sequence_splits,
    config=config,
    target="future_rv_20d",
    verbose=False,
)
summary = summarize_feedback_qrc_result(result)
```

Expected behavior:

- prediction arrays match split lengths;
- feature matrices match split lengths;
- predictions are finite and positive;
- train, validation, and test metrics are finite;
- summary contains model identity, target, split sizes, feature count, and metrics.

## 5. What Tests Should Cover

Initial QRC tests should validate procedure, determinism, and data shape. They should not try to reproduce full Phase 2 notebook metrics.

Recommended test order:

1. `select_temporal_indices` behavior;
2. single-window deterministic feature extraction;
3. feature-matrix construction;
4. tiny synthetic regressor run;
5. summary schema.

## 6. What Tests Should Not Cover Yet

Do not test exploratory notebooks directly.

Do not test full Phase 2 production runs in unit tests. Those are too slow and too dependent on large cached feature files.

Do not test live QPU execution in this branch. Phase 3 hardware work should use a separate controlled validation path.

## 7. Phase 3 Refactor Target

The next cleanup target is to convert the strongest Phase 2 QRC workflow into a single explicit script, for example:

```text
archive/phase2/scripts/run_phase2_feedback_tfim_qrc_reference.py
```

That script should:

1. load the processed Phase 2 modeling frame;
2. create chronological train/validation/test sequence splits;
3. instantiate the documented QRC config;
4. build QRC features;
5. fit the readout on training only;
6. export predictions and metric summaries;
7. write all outputs to deterministic paths under `results/tables/`.

Once that exists, Phase 3 can compare Rydberg/analog reservoirs against a clean digital-QRC reference rather than against notebook state.
