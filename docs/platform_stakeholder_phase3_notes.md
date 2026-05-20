# Platform, stakeholder impact, and Phase 3 planning notes

Date: May 20, 2026

## Quantum platform and resource planning

Phase 2 must specify which qBraid-accessible platforms are intended for Phase 3 and justify why they match the proposed QRC architecture.

## Recommended platform stack

### Primary simulator path

- qBraid-accessible statevector simulators for exact small-reservoir QRC prototypes.
- Density-matrix simulators for small noisy experiments.
- Tensor-network or GPU-accelerated simulators if reservoir size becomes too large for dense statevector simulation.

Purpose:

- establish clean baseline behavior;
- run 7-12 qubit light-touch prototypes;
- evaluate Hamiltonian/encoding/readout variants without hardware noise;
- produce reproducible Phase 2 evidence.

### Primary hardware-aligned path

- TFIM/spin-system QRC implemented first as a simulator model.
- Gate-based approximation can target IBM, IQM, Rigetti, IonQ, or Braket-accessible circuit backends if needed.
- Rydberg/neutral-atom extension can target QuEra-style platforms if the architecture is translated into analog Rydberg/Ising dynamics.

Purpose:

- simulator-first Phase 2;
- targeted QPU validation in Phase 3;
- avoid making hardware availability a Phase 2 bottleneck.

### Error mitigation / noisy-hardware tooling

Potential tools:

- Mitiq;
- mthree;
- zero-noise extrapolation (ZNE).

Use in Phase 3 only if QPU/noisy simulator experiments justify it. Do not make error mitigation a Phase 2 dependency.

## Resource estimates to include in submission

Initial Phase 2 estimates should include:

- qubit count range: target 7-12 qubits for Phase 2 prototype; smaller fallback if runtime is limiting;
- circuit depth / evolution steps: determined by TFIM Trotter steps or equivalent unitary depth;
- shot budgets: exact expectations first; then 512, 2048, optionally 8192 shots for feasibility;
- simulation mode: statevector first, density-matrix for noise pilot;
- classical infrastructure: Python data pipeline, scikit-learn baselines/readouts, qBraid-ready notebooks/scripts;
- outputs: saved metrics tables, figures, and architecture/run configuration.

## Phase 3 execution plan skeleton

1. Freeze public dataset and preprocessing pipeline.
2. Establish classical baselines: persistence/HAR-like, ESN, GARCH-family, optional LSTM.
3. Run simulator TFIM-QRC at fixed architecture on realized-volatility forecasting and transition-risk output.
4. Sweep reservoir size and encoding density.
5. Add shot-budget study.
6. Add noise/density-matrix study.
7. Target small QPU or hardware-relevant backend validation if feasible.
8. Apply error mitigation only where noisy-hardware results justify it.
9. Package final qBraid-executable repository and agent-executable reproducibility workflow.

## Fallback options

If simulator scaling fails:

- reduce qubit count;
- reduce feature dimension;
- use PCA-6 or selected interpretable features;
- reduce Trotter steps / evolution depth;
- use tensor-network/GPU simulator where available.

If QPU access is limited:

- keep Phase 3 validation simulator-first;
- use shot-based simulator as proxy;
- run only one targeted QPU validation experiment.

If TFIM-QRC underperforms:

- test alternative encoding density / re-uploading;
- test RF-QRC/random-unitary reservoir;
- keep regime-transition logic and compare QRC feature maps rather than over-tuning final readout.

If regime-transition labels are unstable:

- use realized-volatility forecasting as required quantitative substrate;
- define transition risk from forecasted volatility path rather than hand-labeled regimes;
- preserve event examples as qualitative stakeholder interpretation.

## Stakeholder impact

Track A stakeholders:

- trading desks;
- market makers;
- risk managers;
- portfolio managers;
- volatility and derivatives desks;
- clearing/risk-control teams.

Value proposition:

- earlier warning of volatility-regime transitions;
- better stress-aware risk limits;
- improved volatility-sensitive portfolio allocation;
- better derivatives pricing and hedging inputs;
- marginal improvements matter because small forecast improvements can affect high-value risk and trading decisions.

Submission framing:

- Do not claim deployable trading alpha.
- Claim a research prototype for volatility-regime transition early warning and volatility-forecasting support.
- Emphasize reproducibility, public data, benchmark discipline, and Phase 3 feasibility.
