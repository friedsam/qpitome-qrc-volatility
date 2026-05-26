# May 25 Milestone 5 — TFIM-QRC Light-Touch Prototype Results

## Purpose

This document records the May 25 light-touch TFIM-QRC prototype results for the qBraid/JonesTrading Phase 2 volatility project.

The milestone goal was not to optimize performance. The goal was to execute the smallest credible end-to-end quantum reservoir computing prototype that substantiates the Phase 2 design claim:

```text
train-only PCA-compressed volatility/VIX features
  -> 40-day anchored temporal input windows
  -> nearest-neighbor TFIM quantum reservoir
  -> exact observable expectation features
  -> ridge readout on log(future_rv_20d)
  -> RMSE / QLIKE / Mincer-Zarnowitz evaluation
```

The stop condition was successful end-to-end execution with interpretable reservoir features and volatility forecasts, even if the model was not yet competitive with classical baselines.

## Prototype configuration

### Fallback prototype

```text
qubits: 6
PCA components: 6
lookback_days: 40
temporal anchors: 6
anchor policy: even
observables: Z
Hamiltonian family: nearest-neighbor TFIM
expectation mode: exact statevector
readout: Ridge(alpha=10) on log(future_rv_20d)
target: future_rv_20d
```

### Primary-small prototype

```text
qubits: 8
PCA components: 8
lookback_days: 40
temporal anchors: 8
anchor policy: even
observables: Z + X + nearest-neighbor ZZ
Hamiltonian family: nearest-neighbor TFIM
expectation mode: exact statevector
readout: Ridge(alpha=10) on log(future_rv_20d)
target: future_rv_20d
```

## PCA diagnostics

The current PCA diagnostics use the 27-feature Phase 2 regression feature set. These replace the older compact/correlation-pruned classifier feature diagnostics.

| PCA components | Cumulative explained variance |
|---:|---:|
| 6 | 0.804892 |
| 8 | 0.874098 |
| 10 | 0.922617 |

Interpretation:

```text
PCA-6: aggressive small-qubit fallback; about 80.5% variance retained.
PCA-8: primary small QRC setting; about 87.4% variance retained.
PCA-10: later sensitivity setting; about 92.3% variance retained.
```

The older approximately 94% PCA-6 number should not be used for this milestone because it belonged to an earlier compact classifier feature set.

## Data shapes

For PCA-6 with 40-day windows:

| Split | X shape | y shape |
|---|---|---|
| train | `(5420, 40, 6)` | `(5420,)` |
| validation | `(1219, 40, 6)` | `(1219,)` |
| test | `(1019, 40, 6)` | `(1019,)` |

These shapes confirm that the prototype used split-local chronological sequence construction with 40-day windows.

## QRC result table

| Run | Qubits | PCA | Anchors | Observables | Reservoir features | Test RMSE | Test QLIKE | Test MZ R² |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| fallback | 6 | 6 | 6 | Z | 6 | 0.107953 | -1.863498 | 0.013055 |
| observable probe | 6 | 6 | 6 | Z + X | 12 | 0.105713 | -1.963174 | 0.037934 |
| observable probe | 6 | 6 | 6 | Z + X + ZZ | 17 | 0.105705 | -1.961164 | 0.038175 |
| primary-small | 8 | 8 | 8 | Z + X + ZZ | 23 | 0.109260 | -1.852447 | 0.008340 |

The best May 25 QRC result by test RMSE was the 6-qubit PCA-6 Z+X+ZZ observable probe.

## Classical comparison

Approximate comparison floors from the May 23 classical baseline milestone:

| Model | Target | Test RMSE | Test QLIKE | Test MZ R² |
|---|---|---:|---:|---:|
| Persistence | `future_rv_20d` | about 0.124 | not recorded here | not recorded here |
| HAR-like ridge / linear | `future_rv_20d` | about 0.101 | not recorded here | about 0.354 |
| PCA-compressed ESN | `future_rv_20d` | about 0.075–0.086 | about -2.44 to -2.49 | about 0.45–0.53 |
| Best May 25 TFIM-QRC | `future_rv_20d` | 0.105705 | -1.961164 | 0.038175 |

Interpretation:

```text
The May 25 TFIM-QRC prototype beats naive persistence by RMSE.
It does not beat the HAR-like classical baseline.
It is far below the PCA-compressed ESN reservoir comparator.
```

This is acceptable for the May 25 milestone because the purpose was light-touch architecture validation, not a performance claim.

## What the prototype proves

The May 25 prototype proves that the project has a runnable, reproducible TFIM-QRC pipeline for the Track A volatility forecasting problem.

Specifically, it demonstrates:

- train-only PCA-compressed volatility/VIX features can be converted into QRC inputs;
- 40-day market memory can be compressed into temporal anchors / virtual nodes;
- a nearest-neighbor TFIM reservoir can process the anchored input sequence;
- exact Z, X, and nearest-neighbor ZZ expectation features can be extracted;
- a classical ridge readout can be trained on log realized volatility;
- the full pipeline can report RMSE, QLIKE, and Mincer-Zarnowitz diagnostics on train, validation, and test splits;
- the QRC model can be compared directly against persistence, HAR-like, and ESN classical baselines.

## What the prototype does not prove

The May 25 prototype does not demonstrate quantum advantage or forecasting superiority.

It does not prove that the current reservoir dynamics capture enough temporal structure for competitive volatility forecasting. The very low Mincer-Zarnowitz R² values indicate weak explanatory calibration. The 8-qubit PCA-8 primary-small run did not improve over the 6-qubit PCA-6 observable-rich run, so immediate qubit/PCA scaling is not justified as the next step.

The current implementation should therefore be treated as viability evidence and a baseline QRC scaffold, not as an optimized QRC model.

## Observed technical lessons

### Observable richness helped modestly

Adding X and ZZ observables improved the 6-qubit test result relative to Z-only:

```text
Z-only test RMSE:      0.107953
Z+X+ZZ test RMSE:     0.105705
Z-only test MZ R²:    0.013055
Z+X+ZZ test MZ R²:   0.038175
```

This suggests that noncomputational and pairwise correlation observables carry some additional signal.

### Scaling to 8 qubits did not help

The 8-qubit PCA-8 Z+X+ZZ run performed worse than the 6-qubit PCA-6 Z+X+ZZ run:

```text
6q / PCA-6 / ZXZZ test RMSE: 0.105705
8q / PCA-8 / ZXZZ test RMSE: 0.109260
```

This suggests that simply adding PCA components and qubits is not the right next move.

### Validation RMSE was not sufficient

The validation RMSE values looked relatively good, but validation and test Mincer-Zarnowitz R² values remained near zero. This indicates that RMSE alone is insufficient for judging model usefulness. Mincer-Zarnowitz calibration diagnostics remain central.

## Phase 3 upgrade direction

The next step should not be a large blind hyperparameter sweep. The code should first be upgraded to determine whether the reservoir features themselves are informative.

Recommended next sequence:

```text
1. Add reservoir feature diagnostics.
2. Add anchor-snapshot / virtual-node observable collection.
3. Add fixed random heterogeneous TFIM parameters.
4. Add residual readout against persistence or HAR-like baselines.
5. Only then run small controlled dynamics sweeps.
```

Initial diagnostics should include:

```text
feature mean / standard deviation
near-constant feature count
feature correlation matrix
effective rank / singular spectrum
condition number
train / validation / test feature distribution shift
correlation of each reservoir feature with the target
```

The highest-value architecture upgrade is likely anchor-snapshot feature collection: instead of measuring observables only after the final anchor, measure after each temporal anchor and concatenate the observable features. For the 6-qubit Z+X+ZZ setting this changes the readout feature count from 17 to 102, while keeping the reservoir small and interpretable.

## Sign-off statement

May 25 Milestone 5 is complete.

The TFIM-QRC prototype runs end-to-end, produces interpretable reservoir features and volatility forecasts, and can be compared against classical volatility baselines. The results support the Phase 2 architecture and resource-justification narrative, while clearly showing that further architecture work is needed before making any performance or advantage claim.
