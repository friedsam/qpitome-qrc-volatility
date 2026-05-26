# May 24 Milestone 4 — QRC Architecture Design Specification

## Purpose

This document specifies the first QRC architecture for the Phase 2 volatility-regime forecasting project.

The goal is not to declare a final optimal QRC design. The goal is to define a primary, literature-grounded architecture and a small set of controlled probes so that Phase 3 implementation becomes systematic rather than design guessing.

The architecture is informed by:

- the Phase 2 challenge design space;
- the Phase 1 QRC proposal;
- QRC literature on spin reservoirs, memory, stability, measurement, and finite sampling;
- the May 23 classical baseline results, especially the PCA-compressed ESN regression baseline.

The main QRC target is continuous realized-volatility forecasting:

```text
primary target:
  future_rv_20d

secondary target:
  future_rv_5d

primary metrics:
  RMSE
  QLIKE
  Mincer-Zarnowitz regression statistics
```

A volatility-regime transition warning layer can later be derived from the predicted volatility path, but the QRC itself is first specified as a realized-volatility regression model.

## Design constraints inherited from May 23

The May 23 baseline work changed the QRC benchmark target.

The QRC prototype should not merely beat naive persistence. It should be compared against a meaningful classical reservoir control.

Key May 23 findings:

```text
Persistence:
  useful primitive volatility-clustering floor

HAR / Ridge / ElasticNet:
  strong low-cost classical regression floor

PCA-compressed ESN:
  sophisticated classical reservoir comparator
  40-day memory improved 20-day volatility forecasting
  log-volatility ridge readout stabilized QLIKE
```

Design consequences:

```text
QRC should use PCA-compressed volatility/VIX inputs.
QRC should explicitly represent 40-day memory or a compressed approximation to it.
QRC should use a ridge-style classical readout.
QRC should predict log-volatility and transform back before evaluation.
QRC should be compared to PCA-compressed ESN, not only to persistence/HAR.
```

## Primary architecture overview

The primary QRC prototype is a transverse-field Ising / spin-system quantum reservoir.

```text
Input:
  PCA-compressed volatility/VIX state from SPY+VIX feature pipeline

Reservoir:
  fixed TFIM / spin-system dynamics

Memory:
  compressed temporal injection from a 40-day rolling window

Readout features:
  expectation values of measured observables

Classical head:
  ridge regression on log(future_rv_20d)

Prediction:
  exp(readout output) -> realized-volatility forecast
```

High-level architecture:

```text
SPY+VIX engineered features
        |
        v
Train-only scaler + PCA
        |
        v
40-day rolling window
        |
        v
Temporal anchor selection / virtual nodes
        |
        v
Angle encoding into TFIM quantum reservoir
        |
        v
Reservoir evolution
        |
        v
Observable expectations: Z, X, ZZ
        |
        v
Ridge readout on log-volatility
        |
        v
future_rv_20d forecast
        |
        v
RMSE / QLIKE / Mincer-Zarnowitz evaluation
```

## Hamiltonian / unitary

### Primary Hamiltonian family

Use a transverse-field Ising model:

```text
H = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + Σ_i g_i X_i
```

where:

```text
Z_i:
  local longitudinal field / encoded market-state memory

Z_i Z_j:
  pairwise spin interactions / nonlinear feature mixing

X_i:
  transverse field / noncommuting dynamics
```

### Topology variants

The primary implementation should start with sparse or nearest-neighbor coupling for simulator/gate feasibility, then compare against a denser version if resources allow.

| Variant | Role | Rationale |
|---|---|---|
| nearest-neighbor TFIM | first implementation | simplest, hardware-aware, easier to debug |
| random sparse TFIM | first extension | richer mixing without full all-to-all cost |
| fully connected TFIM | simulator comparison | closer to realized-volatility QRC literature precedent |
| analog/Rydberg-style reservoir | Phase 3 extension | closer to large-scale analog QRC literature and neutral-atom hardware relevance |

### Evolution block

One reservoir step consists of:

```text
1. input encoding rotations
2. fixed TFIM evolution for time Δt or one/two Trotter blocks
3. optional measurement-feature extraction at selected virtual nodes
```

Initial implementation should use shallow Trotterized evolution:

```text
fallback:
  1 Trotter step per temporal anchor

primary:
  2 Trotter steps per temporal anchor

extension:
  3+ Trotter steps / swept evolution time
```

The exact evolution time and coupling scale are experimental parameters because QRC memory depends on reservoir timescale. Too little evolution may under-mix inputs; too much evolution may scramble useful fading memory.

## Input features and scaling

### Feature source

Use the existing Phase 2 SPY+VIX feature pipeline.

The full engineered feature set contains volatility, return, range, drawdown, and VIX state variables. For QRC, the first input should be PCA-compressed because:

- features are highly correlated;
- PCA-compressed ESN worked well;
- near-term QRC has limited qubit/input bandwidth;
- PCA provides a fair comparison to the compressed ESN baseline.

### PCA settings

Use train-only scaling and train-only PCA.

```text
fit StandardScaler on train split only
fit PCA on train split only
apply fixed scaler/PCA to validation and test
```

Primary settings:

| PCA dimension | Role |
|---:|---|
| 6 | minimal compressed QRC baseline |
| 8 | primary QRC input setting |
| 10 | sensitivity / larger-input extension |

Recommended first QRC input:

```text
PCA-8 volatility/VIX state
```

Reason:

```text
PCA-6 is maximally compact.
PCA-8 preserves more information while still being plausible for 8–10 qubit prototypes.
PCA-10 is useful as a sensitivity setting but may exceed the smallest near-term design.
```

## Input encoding

### First implementation: angle encoding

Use bounded angle encoding into single-qubit rotations.

For a PCA vector `x_t`, scale each component to a bounded interval:

```text
x_j -> θ_j in [-θ_max, θ_max]
```

with a first choice:

```text
θ_max = π/2
```

Then encode with rotations such as:

```text
Ry(θ_j)
```

or a two-axis variant:

```text
Rz(φ_j) Ry(θ_j)
```

### Encoding policy

| Encoding | Role | Reason |
|---|---|---|
| Ry angle encoding | first implementation | simplest, transparent, easy to debug |
| Rz + Ry encoding | first extension | richer phase/amplitude mixing |
| feature re-uploading | primary temporal mechanism | allows limited qubits to process multiple time anchors |
| amplitude encoding | Phase 3 extension | compact but state-preparation overhead is nontrivial |

### Feature-to-qubit mapping

For PCA dimension `k` and qubit count `n`:

```text
if k <= n:
  map one PCA component to one qubit

if k > n:
  cycle/re-upload components or use multiple encoding sublayers
```

Primary setting:

```text
PCA-8 with 8 qubits:
  one PCA component per qubit
```

Fallback setting:

```text
PCA-6 with 6 qubits:
  one PCA component per qubit
```

Extension:

```text
PCA-10 with 8 qubits:
  re-upload remaining components or use two encoding layers
```

## Memory and temporal injection

The ESN baseline shows that 40-day memory matters for `future_rv_20d`. QRC should therefore be designed around a 40-day rolling market window.

A naive full design would inject all 40 daily PCA vectors sequentially. This is conceptually clean but may be expensive. The primary design should therefore use compressed temporal anchor nodes first.

### Temporal anchor policy

For each sample, start from the 40-day window ending at date `t`:

```text
[x_{t-39}, ..., x_t]
```

Use anchor days such as:

```text
8-anchor version:
  t-39, t-34, t-29, t-24, t-19, t-14, t-7, t

10-anchor version:
  t-39, t-35, t-31, t-27, t-23, t-19, t-15, t-10, t-5, t
```

Alternative anchor schedules can emphasize recency:

```text
t-39, t-29, t-19, t-14, t-9, t-5, t-2, t
```

### Memory probes

| Probe | Description | Role |
|---|---|---|
| 6 anchors | low-cost compressed memory | fallback |
| 8 anchors | primary first design | balances memory and circuit cost |
| 10 anchors | richer memory | extension |
| full 40-step injection | closest ESN analogue | simulator-only extension if feasible |

### Reset policy

Primary policy:

```text
reset per 40-day sample/window
```

Reason:

- clean supervised-learning setup;
- avoids leakage across samples;
- easier batching and comparison to ESN sequence samples.

Phase 3 probe:

```text
state carryover within chronological split only
reset at train/validation/test boundaries
```

Reason:

- closer to recurrent reservoir dynamics;
- may improve memory;
- requires careful leakage and stability controls.

### Feedback policy

Initial design:

```text
no feedback
```

Phase 3 probe:

```text
optional feedback/reinjection of prior measured observables or prior forecast residuals
```

Feedback is deferred because it complicates leakage, stability, and interpretability.

## Readout observables

Measured observables become reservoir-state features for the classical readout.

Use exact expectation values first, then finite-shot estimates later.

### Observable ladder

| Stage | Observables | Purpose |
|---|---|---|
| 1 | `<Z_i>` | simplest computational-basis readout |
| 2 | `<Z_i>`, `<X_i>` | include noncommuting transverse-field information |
| 3 | `<Z_i>`, `<X_i>`, `<Z_i Z_{i+1}>` | include pairwise interaction/correlation features |
| 4 | selected random Pauli strings / selected two-point correlators | richer feature map if needed |

First implementation:

```text
Z-only readout
```

Primary comparison:

```text
Z + X + nearest-neighbor ZZ
```

Reason:

```text
Z is easiest.
X may expose transverse-field dynamics.
ZZ correlators expose interaction structure and quantum/classical correlation features.
```

## Classical head and output

Use a ridge regression readout trained on log-volatility.

Training target:

```text
y_train_head = log(max(future_rv_20d, eps))
```

Prediction:

```text
raw_head_output = ridge(observable_features)
y_pred = exp(raw_head_output)
```

Evaluation:

```text
RMSE(y_true, y_pred)
QLIKE(y_true, y_pred)
Mincer-Zarnowitz(y_true, y_pred)
```

Reason:

The May 23 ESN baseline showed that raw-volatility prediction can produce near-zero forecasts that look acceptable by RMSE but fail badly under QLIKE. Log-volatility readout enforces positive forecasts and improves calibration stability.

## Hybrid integration

The QRC reservoir is not the final predictor by itself. It is a feature generator for a classical volatility-forecasting head.

Pipeline:

```text
1. Build train/validation/test chronological splits.
2. Fit scaler/PCA on train only.
3. Build 40-day rolling windows.
4. Select temporal anchors.
5. Encode anchors into TFIM reservoir.
6. Collect expectation features.
7. Fit ridge readout on train only.
8. Select hyperparameters using validation only.
9. Report train/validation/test metrics.
```

Comparison groups:

```text
naive persistence
HAR-like linear/ridge
full-feature Ridge / ElasticNet
PCA-compressed ESN
PCA-compressed TFIM-QRC
```

## Resource estimate

Initial resource table:

| Setting | Qubits | PCA dim | Temporal anchors | Evolution depth | Observables | Mode | Role |
|---|---:|---:|---:|---:|---|---|---|
| fallback | 6 | 6 | 6 | 1 | Z | exact | cheapest validation |
| primary-small | 8 | 8 | 8 | 1–2 | Z, X | exact | first serious QRC |
| primary-rich | 8 | 8 | 8–10 | 2 | Z, X, ZZ | exact | main Phase 3 target |
| extended | 10 | 8–10 | 10 | 2–3 | Z, X, ZZ | exact | richer simulator study |
| high-resource | 12 | 8–10 | 10–20 | 2–3 | Z, X, ZZ, selected Pauli | exact / shots | Phase 3 scaling |

Shot budget plan:

| Mode | Shots | Role |
|---|---:|---|
| exact expectations | none | first implementation and architecture debugging |
| low-shot | 512 | feasibility lower bound |
| medium-shot | 2048 | practical benchmark |
| high-shot | 8192 | closer to stable readout features |

Noise plan:

| Noise model | Role |
|---|---|
| none | first exact simulator baseline |
| depolarizing | generic gate/readout degradation probe |
| amplitude damping | relaxation / energy-loss probe |
| shot noise | finite-sampling resource estimate |

## Config schema

The QRC experiments should be driven by a small configuration object rather than hard-coded notebooks.

Proposed schema:

```yaml
experiment_name: qrc_tfim_pca8_seq40_anchor8

data:
  processed_path: data/processed/phase2_spy_vix_volatility.csv
  target: future_rv_20d
  feature_set: phase2_full_features
  scaler: standard
  pca_components: 8
  train_end: 2014-12-31
  val_end: 2019-12-31

memory:
  lookback_days: 40
  anchor_count: 8
  anchor_policy: evenly_spaced
  reset_policy: reset_per_window
  feedback: false

encoding:
  type: angle
  rotations: [ry]
  angle_max: 1.57079632679
  clipping: true
  reuploading: temporal_anchors

reservoir:
  family: tfim
  qubits: 8
  topology: nearest_neighbor
  longitudinal_field: trainable_no
  transverse_field: fixed_random
  coupling_scale: 1.0
  field_scale: 1.0
  evolution_time: 1.0
  trotter_steps_per_anchor: 2
  seed: 42

readout:
  observables: [z, x, zz_nearest]
  expectation_mode: exact
  shots: null
  model: ridge
  ridge_alpha: 10.0
  target_transform: log

metrics:
  - rmse
  - qlike
  - mincer_zarnowitz

comparators:
  - persistence
  - har_ridge
  - full_feature_elasticnet
  - pca_esn_log_target
```

## Controlled experiment plan

Do not implement multiple QRC families first. Implement one TFIM-QRC and vary only a few axes.

### Phase 3 first implementation

```text
1. Build QRC feature-generation code for exact expectations.
2. Implement fallback 6-qubit PCA-6, 6-anchor Z-only run.
3. Validate shapes, leakage control, and metric pipeline.
4. Move to 8-qubit PCA-8, 8-anchor Z/X/ZZ run.
5. Compare against PCA-compressed ESN.
```

### First controlled probes

| Probe | Values | Reason |
|---|---|---|
| PCA dimension | 6, 8 | compare minimal vs primary compressed inputs |
| qubits | 6, 8 | test smallest feasible reservoirs |
| anchor count | 6, 8 | test compressed 40-day memory |
| observables | Z; Z+X; Z+X+ZZ | test readout richness |
| Trotter steps | 1, 2 | test under-mixing vs cost |
| ridge alpha | 1, 10 | align with ESN readout behavior |

### Deferred probes

| Probe | Reason for deferral |
|---|---|
| full 40-step injection | may be costly; use after anchor design works |
| fully connected TFIM | denser circuits; use as simulator comparison |
| carryover state | leakage/stability risk; use only after reset-per-window baseline |
| finite-shot mode | use after exact expectations establish signal |
| noise models | resource/stability Phase 3 layer |
| RF-QRC | fallback if recurrent QRC is unstable or too slow |
| onion/parallel QRC | multi-timescale extension, not first implementation |
| amplitude encoding | state-preparation overhead; compare after angle encoding works |
| LSTM/GRU/GARCH | classical Phase 3 expansion, not QRC architecture dependency |

## Architecture diagram draft

A diagram for the paper should show the following blocks:

```text
[SPY + VIX daily data]
        |
        v
[Engineered volatility / VIX / return features]
        |
        v
[Train-only StandardScaler + PCA]
        |
        v
[40-day rolling window]
        |
        v
[Temporal anchor selection]
        |
        v
[Angle encoding / re-uploading]
        |
        v
[Fixed TFIM quantum reservoir]
        |
        v
[Observable expectations: Z, X, ZZ]
        |
        v
[Ridge readout on log-volatility]
        |
        v
[future_rv_20d forecast]
        |
        v
[RMSE / QLIKE / Mincer-Zarnowitz]
```

The diagram should visually emphasize that only the classical readout is trained. The reservoir Hamiltonian is fixed for each configuration.

## Subsequent actions

### Immediate next actions

```text
1. Convert this architecture note into one paper-ready figure.
2. Create the QRC config schema as a YAML or dataclass.
3. Implement exact-expectation TFIM-QRC feature generation.
4. Start with fallback 6-qubit PCA-6, 6-anchor, Z-only run.
5. Validate against the existing metrics pipeline.
```

### Next design checks

```text
1. Confirm which Hamiltonian topology is fastest to implement in Qiskit/qBraid.
2. Decide whether first prototype uses statevector expectations or circuit sampling.
3. Define the anchor-selection function.
4. Define observable list and feature naming convention.
5. Ensure train-only PCA and chronological splits are reused exactly from May 23.
```

### Phase 2 paper use

This document supports the following paper claims:

```text
1. The QRC design is not arbitrary; it is grounded in QRC literature and the challenge design space.
2. The QRC benchmark is strengthened by the May 23 PCA-compressed ESN result.
3. The first QRC prototype is intentionally narrow: TFIM first, exact expectations first, ridge readout first.
4. Extensions are defined as controlled probes rather than broad architecture sprawl.
```

## Milestone sign-off condition

May 24 is complete when:

```text
A judge can read this architecture note and understand:
  what QRC system we intend to build;
  how data enters the quantum reservoir;
  where memory comes from;
  what is measured;
  how the readout is trained;
  what resources are expected;
  how the prototype will be compared to classical baselines;
  what Phase 3 probes are planned.
```
