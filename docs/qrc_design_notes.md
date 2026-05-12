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