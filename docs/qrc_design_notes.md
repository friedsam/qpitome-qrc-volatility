# QRC Design Notes

## Purpose

Track the evolution of the Quantum Reservoir Computing prototype for the May 8 milestone.

The document should record implemented versions, observed behavior, and design decisions. It is not a full project plan.

## Milestone Context

May 8 target:

```text
minimal QRC prototype
- tiny QRC simulator pipeline runs
- QRC concept understood well enough to modify
```

## Version Log

### v0.1 — Minimal synthetic QRC prototype

**Date:** 2026-05-12  
**Script:** `scripts/run_qrc_tiny.py`  
**Status:** Runs successfully.

This version tests the smallest end-to-end QRC path:

```text
synthetic input features
→ Qiskit reservoir circuit
→ statevector expectation values
→ quantum feature matrix
→ logistic-regression readout
```

Configuration:

```text
n_samples = 300
n_qubits  = 4
n_layers  = 2
backend   = Qiskit Statevector expectation values
readout   = LogisticRegression(class_weight="balanced")
observables = single-qubit Z expectations
```

Observed output:

```text
Phi_train shape: (210, 4)
Phi_test shape:  (90, 4)

accuracy:          0.744
balanced accuracy: 0.744
```

Interpretation:

This confirms that the QRC circuit-to-feature-to-readout pipeline works technically.

The score is not evidence of predictive value for volatility because the data are synthetic.

Current limitations:

```text
- synthetic data only
- no real volatility windows yet
- no baseline comparison in this script yet
- exact statevector expectations only
- no shot-based simulation yet
- no qBraid/cloud backend yet
```

## Current Design

### Encoding

Each input feature is mapped to qubit rotations:

```text
x_i → RX(x_i), RZ(0.5 x_i)
```

### Reservoir

The reservoir is fixed, not trained.

Each layer applies random single-qubit rotations followed by ring entanglement:

```text
RY(random), RZ(random), CX ring
```

### Measurement

The current feature vector consists of single-qubit Z expectation values:

```text
<Z0>, <Z1>, <Z2>, <Z3>
```

### Readout

Only the classical readout is trained:

```text
LogisticRegression(class_weight="balanced")
```

## Open Next Step

The next version should make the prototype more relevant without overbuilding. Candidate directions:

```text
- add trivial/raw-feature baselines
- add richer observables such as ZZ correlations
- connect to the real volatility-window data loader
- refactor reusable QRC code into src/
```

The next choice should depend on which piece becomes the bottleneck first.

## v0.2 — Real-data QRC diagnostic

**Date:** 2026-05-12  
**Script:** `scripts/run_qrc_real_tiny.py`  
**Status:** Runs successfully, but QRC-only performance is weak.

This version connects the QRC prototype to the same processed market-stress dataset and chronological train/validation/test split used by the ESN baseline.

Pipeline:

```text
20 × 12 market-stress sequence
→ flatten to 240 features
→ StandardScaler + PCA(4)
→ 4-qubit QRC reservoir
→ Z and nearest-neighbor ZZ expectation features
→ LogisticRegression readout
```

Configuration:

```text
n_qubits = 4
n_layers = 2
PCA components = 4
QRC observables = 8
backend = Qiskit Statevector expectation values
readout = LogisticRegression(class_weight="balanced")
```

Data split:

```text
train: 1993-03-26 → 2015-12-31, n = 5732
val:   2016-01-04 → 2019-12-31, n = 1006
test:  2020-01-02 → 2024-04-08, n = 1073
```

Main test results:

```text
Majority baseline:
balanced_accuracy = 0.500
PR-AUC            = 0.249

PCA-raw LogisticRegression:
balanced_accuracy = 0.738
PR-AUC            = 0.550
confusion matrix  = [[484, 322],
                     [ 33, 234]]

QRC-only LogisticRegression:
balanced_accuracy = 0.524
PR-AUC            = 0.215
confusion matrix  = [[235, 571],
                     [ 65, 202]]

PCA+QRC LogisticRegression:
balanced_accuracy = 0.740
PR-AUC            = 0.560
confusion matrix  = [[514, 292],
                     [ 42, 225]]
```

Interpretation:

QRC-only is not useful in this configuration. It performs near chance by balanced accuracy and produces many false positives.

The PCA-compressed linear baseline is much stronger than QRC-only. Adding QRC features to PCA gives only a small improvement: slightly higher balanced accuracy and PR-AUC, and fewer false positives than PCA alone, but the gain is marginal.

Current conclusion:

```text
The real-data QRC pipeline works technically.
The current shallow random QRC is not yet a competitive model.
PCA compression plus a linear readout is the stronger baseline.
Future QRC work should focus on controlled ablations, not blind reservoir tweaking.
```