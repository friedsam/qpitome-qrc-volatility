# QRC Literature-to-Design Map

## Purpose

This note preserves the architecture-relevant ideas extracted from the challenge resources, QRC literature, and the May 23 classical baseline work.

It is separate from the May 24 architecture specification. Its role is to make the design reasoning easier to scan and reuse when building the Phase 2 paper and Phase 3 prototype.

The central question is:

```text
Which parts of the QRC architecture are supported by the literature, which parts are driven by our own baseline results, and which parts should become controlled Phase 3 probes?
```

## One-minute keyword index

| Keyword | Relevant source(s) | Useful design idea |
|---|---|---|
| `TFIM` | Li et al.; Kutvonen/Fujii/Sagawa; challenge description | First Hamiltonian family for realized-volatility QRC. |
| `REALIZED_VOLATILITY` | Li et al.; challenge description; May 23 baselines | Continuous volatility regression is the right quantitative substrate. |
| `RIDGE_READOUT` | QRC challenge; ESN baseline; reservoir-computing literature | Fixed reservoir with trained linear/ridge readout. |
| `PCA_INPUTS` | Classical baseline/PCA report; May 23 ESN | Compression is both hardware-aware and empirically useful. |
| `40_DAY_MEMORY` | May 23 ESN | Main empirical memory target for QRC probes. |
| `TEMPORAL_MULTIPLEXING` | Challenge description; QRC literature | Sequential/virtual-node input injection into a fixed reservoir. |
| `OBSERVABLES` | Spin/QRC literature; challenge description | Reservoir features from measured expectations such as Z, X, ZZ. |
| `NOISE` | Ahmed et al.; finite-shot QRC paper; challenge description | Phase 3 shot/noise robustness and possible noise-as-regularization. |
| `FINITE_SHOTS` | Finite-sampled QRC paper; challenge resource planning | Exact expectations first, finite-shot studies later. |
| `RF_QRC` | Ahmed et al.; finite-sampled QRC paper | Parallel/non-recurrent fallback if recurrent QRC is unstable or too expensive. |
| `PARALLEL_RESERVOIRS` | Tandon et al. onion QRC | Phase 3 extension: multiple small reservoirs for different memory scales. |
| `AMPLITUDE_ENCODING` | Challenge description; financial QML/QRC papers | Possible extension; not first implementation because of state-preparation overhead. |
| `ANGLE_ENCODING` | Challenge description; near-term circuit practice | First implementation because it is simple, transparent, and debug-friendly. |
| `RESET_POLICY` | Reservoir stability literature; leakage control | Reset per sample first; carryover only as a controlled probe. |
| `PHASE3_EXTENSION` | All sources | Larger sweeps, GARCH/LSTM, noise, RF-QRC, onion QRC, hardware-oriented designs. |

## Core synthesis

The QRC design should be framed as a literature-grounded experimental architecture, not a single final circuit.

The primary design direction is:

```text
Primary QRC architecture:
  TFIM / spin-system quantum reservoir

Primary empirical anchor:
  PCA-compressed ESN with 40-day memory and log-volatility ridge readout

Primary literature anchor:
  fixed quantum reservoir + temporal input injection + measured observables + trained classical readout
```

This architecture is not merely a quantum copy of the ESN. It is a convergence of:

1. the challenge QRC design space;
2. realized-volatility QRC literature;
3. general QRC memory/stability literature;
4. the May 23 result showing that PCA-compressed reservoir memory is useful for `future_rv_20d`.

## Literature-to-design extraction table

| Source | What it contributes | Useful design implication for our QRC | Possible limitations |
|---|---|---|---|
| Challenge description | Defines QRC as near-term temporal learning with a fixed reservoir and trained readout. It explicitly asks for Hamiltonian, encoding, measurement, feedback, hybrid architecture, resource, and benchmark choices. | The architecture document must specify Hamiltonian, input encoding, memory policy, readout observables, hybrid classical head, qubit count, shots/noise, and reproducibility path. | The challenge provides a broad design space, not one optimal architecture. |
| Phase 1 submission | Already commits to rolling market windows, measured observables as nonlinear features, a trained classical readout, and comparison against persistence, linear, and compact nonlinear classical baselines. | May 24 should refine this commitment rather than invent an unrelated model. | Phase 1 was conceptual; Phase 2 needs more precise implementation fields. |
| Kornjača et al., large-scale analog QRC | Demonstrates scalable, gradient-free quantum reservoir learning on neutral-atom hardware with preprocessing, reservoir evolution, measurement outputs, and classical postprocessing. | Supports the reservoir-as-feature-generator pattern and motivates analog/Rydberg QRC as a Phase 3 hardware-relevant extension. | Their encoding and hardware are analog neutral-atom specific; not the immediate gate/simulator-first prototype. |
| Li et al., QRC for realized-volatility forecasting | Directly relevant finance paper using QRC for realized-volatility forecasting. Uses a transverse-field Ising reservoir and hardware-aware feature selection/compression. | Strongest support for choosing TFIM/spin-system QRC first and for treating realized-volatility regression as the core task. Also supports feature compression as a hardware-aware necessity. | Dataset, feature set, target scaling, and split may differ from ours; avoid direct numerical comparison. |
| Kutvonen/Fujii/Sagawa, optimizing QRC memory | Studies QRC memory capacity in transverse-field Ising systems. Shows memory depends on interaction structure, magnetic coupling, and input interval/timescale. | Treat QRC design as a memory-engineering problem. Probe evolution time, coupling strength, topology, and input interval rather than assuming arbitrary TFIM dynamics will work. | Memory-capacity tasks are not the same as realized-volatility forecasting. |
| Zhu et al., practical few-atom QRC | Supports small-system QRC feasibility and measurement-aware practical design. | Justifies starting with 6–12 qubit prototypes and thinking carefully about what observables are actually measurable. | Physical implementation differs from our likely gate-model/simulator prototype. |
| Tandon et al., onion QRC for corrosion | Uses multiple smaller reservoirs operating at different time scales. | Strong Phase 3 extension: parallel small QRC reservoirs for short/medium/long volatility memory instead of one large deep circuit. | Extension only; too much for the May 24 primary design. |
| Ahmed et al., robust QRC / generalized synchronization / stability | Frames recurrent QRC as a dynamical system and connects stability/fading memory to useful reservoir behavior. Discusses robustness and dissipation/noise. | Add stability diagnostics: perturbation sensitivity, input-memory decay, seed/initial-state sensitivity, and noise-as-regularization probes. | General chaotic forecasting theory; not specific to SPY/VIX realized volatility. |
| Finite-sampled QRC training paper | Studies finite-shot sampling effects and denoising/SVD-style stabilization of reservoir activation matrices. | Exact expectations first; later evaluate 512/2048/8192 shots, ridge/SVD denoising, and recurrent QRC versus RF-QRC under shot noise. | Mostly Phase 3 resource planning; not required for the first exact-simulator prototype. |
| Financial QLSTM/QRC papers | Emphasize nonstationarity, heavy tails, regime shifts, lag embeddings, and qubit-constrained encodings for financial series. | Support testing amplitude or lag-compressed encodings later and comparing against LSTM/GRU in Phase 3. | Broader finance sequence modeling; not the immediate QRC architecture. |
| Classical baseline/PCA report | Shows high collinearity among VIX/RV features, PCA-6 preserving most compact-feature variance, and VIX/RV variables dominating feature importance. | PCA-6/8/10 are data-driven compression choices, not arbitrary qubit workarounds. Feed volatility/VIX state first. | Earlier report used binary high-volatility classification, so it supports feature diagnostics rather than the final regression metric. |
| May 23 ESN result | PCA-compressed log-target ESN with 40-day memory substantially improved `future_rv_20d` test performance over persistence and HAR-like baselines. | Strongest project-specific QRC design signal: use PCA-compressed inputs, 40-day memory, reservoir dynamics, ridge readout, and log-volatility targets. | Internal bounded experiment, not a final exhaustive benchmark. |

## Design conclusions

### 1. TFIM/spin-system QRC is the right first architecture

The primary Hamiltonian family should be a transverse-field Ising or spin-system reservoir:

```text
H = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + Σ_i g_i X_i
```

This is supported by the realized-volatility QRC literature and by QRC memory-capacity studies. It is also explicitly within the challenge design space.

Recommended variants:

```text
Primary prototype:
  sparse or nearest-neighbor TFIM for gate/simulator feasibility

Simulator comparison:
  fully connected TFIM, because this appears in realized-volatility QRC literature

Phase 3 extension:
  random sparse spin reservoirs and analog/Rydberg reservoirs
```

### 2. QRC should be a reservoir analogue of the successful ESN

The May 23 result tells us which classical reservoir ingredients worked:

```text
PCA-compressed input
40-day sequence memory
nonlinear reservoir transformation
ridge readout
log-volatility target
```

The QRC should therefore use:

```text
input:
  PCA-6 or PCA-8 volatility/VIX state

memory:
  derived from a 40-day rolling window

reservoir:
  TFIM quantum dynamics

readout:
  measured observables -> ridge regression on log(future_rv_20d)
```

This enables a clean comparison:

```text
ESN:
  classical random nonlinear reservoir + ridge log-volatility readout

QRC:
  quantum spin reservoir + ridge log-volatility readout
```

### 3. Memory is the central design axis

Memory is the point where the literature and our empirical results align most strongly.

The QRC design should explicitly probe:

```text
input interval / evolution time
number of temporal injection nodes
40-day window compression
coupling strength
topology
reset versus carryover
```

Recommended memory probes:

```text
Probe A:
  8 anchor nodes from the 40-day window

Probe B:
  10 anchor nodes from the 40-day window

Probe C:
  full 40-step sequential injection if simulator cost allows

Probe D:
  RF-QRC / recurrence-free feature map if recurrent propagation is unstable or too noisy
```

### 4. PCA compression is both empirical and hardware-aware

PCA should be described as both:

1. a hardware-aware compression strategy for limited qubit/input bandwidth;
2. an empirically justified representation because volatility/VIX features are highly correlated and ESN performance survives compression.

Recommended PCA settings:

```text
PCA-6:
  smallest compressed input, good QRC-comparable baseline

PCA-8:
  primary Phase 3 setting, balancing information and qubit count

PCA-10:
  sensitivity/extension setting if resources allow
```

### 5. Angle encoding first; amplitude encoding later

The first implementation should use angle encoding:

```text
PCA component -> bounded Ry/Rz rotation angle
```

Reason:

```text
simple
transparent
easy to debug
compatible with sequential re-uploading
low state-preparation overhead
```

Amplitude encoding remains a serious extension because it can encode more information per qubit, but it introduces state-preparation overhead and is less convenient for the first Phase 3 prototype.

### 6. Readout observables should be staged

Recommended observable ladder:

```text
Stage 1:
  <Z_i>

Stage 2:
  <Z_i>, <X_i>

Stage 3:
  <Z_i>, <X_i>, <Z_i Z_{i+1}>

Stage 4:
  selected two-point correlators / random Pauli observables
```

Reasoning:

```text
Z observables:
  easiest computational-basis measurements

X observables:
  useful because TFIM includes noncommuting X dynamics

ZZ observables:
  expose interaction/correlation structure
```

### 7. Reset policy should be experimental

There are two legitimate reset policies:

```text
reset per sample/window:
  cleaner supervised-learning setup
  easier batching
  avoids leakage across examples

state carryover:
  closer to recurrent reservoir dynamics
  may improve memory
  higher leakage/stability risk
```

Recommended design:

```text
Primary:
  reset per 40-day sample/window

Phase 3 probe:
  carryover within each chronological split only,
  reset at train/validation/test boundaries
```

## Recommended primary QRC specification

```text
Hamiltonian:
  sparse/nearest-neighbor TFIM, with fully connected TFIM as simulator comparison

Input:
  PCA-8 volatility/VIX state, train-only scaler/PCA

Memory:
  40-day rolling window
  first implementation uses 8–10 temporal anchor nodes
  full 40-step sequential injection as extension

Encoding:
  angle encoding via Ry/Rz rotations
  scaled PCA features clipped to bounded angle range

Reservoir evolution:
  encode anchor-day PCA vector
  evolve under fixed TFIM block
  repeat for anchor nodes / virtual nodes

Reset:
  reset per sample/window first
  carryover only as controlled Phase 3 probe

Readout:
  exact expectation values first
  <Z_i> baseline
  <Z_i>, <X_i>, <Z_i Z_{i+1}> extensions

Classical head:
  ridge regression on log(future_rv_20d)
  exp-transform predictions before RMSE/QLIKE/MZ evaluation

Metrics:
  RMSE
  QLIKE
  Mincer-Zarnowitz

Classical comparators:
  persistence
  HAR/Ridge/ElasticNet
  PCA-compressed ESN

Resource probe:
  qubits = 6, 8, 10, 12
  exact expectations first
  shots = 512, 2048, 8192 later
  noise = depolarizing + amplitude damping later
```

## Critical Phase 3 experimental probes

| Probe | Why it matters | Expected insight |
|---|---|---|
| Qubit count 6/8/10/12 | Challenge expects resource scaling. | Whether additional Hilbert-space dimension improves reservoir features. |
| PCA dimension 6/8/10 | Controls information retained versus input/resource cost. | Whether QRC needs richer input or benefits from compression. |
| Memory anchor count 6/8/10/full 40 | ESN says 40-day memory matters; QRC cannot naively process everything without cost. | Whether compressed temporal multiplexing preserves useful memory. |
| Evolution time / Trotter depth | Memory literature says timescale affects reservoir memory. | Identify under-mixed versus over-scrambled dynamics. |
| Coupling topology | Fully connected TFIM has literature precedent; sparse TFIM is hardware-aware. | Accuracy/resource tradeoff. |
| Observable set | Z-only may underuse quantum correlations. | Whether X/ZZ/correlators add predictive value. |
| Reset versus carryover | Recurrent reservoirs can carry memory but risk instability/leakage. | Whether true recurrence improves forecasts safely. |
| Exact versus finite shots | Required for resource planning. | Shot budget and robustness of measured features. |
| Noise/dissipation | Robust-QRC literature suggests noise can sometimes regularize. | Whether moderate noise hurts or helps volatility forecasting. |
| RF-QRC | Parallelizable and avoids recurrent noise propagation. | Useful fallback if recurrent QRC is unstable or too costly. |
| Onion / parallel reservoirs | Multi-timescale reservoir structure. | Potential way to model short/medium/long volatility memory without one large circuit. |

## Immediate use in May 24

This literature map should feed two documents:

```text
docs/qrc_architecture_design.md
  concise implementation-facing architecture specification

docs/qrc_literature_design_map.md
  deeper rationale and paper-to-design traceability
```

The architecture design document should be shorter and more execution-oriented. This literature map should remain the reference for why the design choices are reasonable and which extensions are motivated by which sources.
