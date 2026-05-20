# Theoretical and analytical justification notes

Date: May 20, 2026

Status: working notes for Phase 2 submission. These notes convert the challenge rubric into concrete design hypotheses and lightweight prototype tests.

## Core framing

The target problem should be framed as volatility-regime transition early warning, not merely scalar volatility prediction and not the older high-stress binary classifier.

Recommended headline sub-problem:

- Forecast calm-to-turbulent volatility-regime transition risk over a fixed lead horizon using public equity-market data.

Supporting substrate:

- Multi-horizon realized-volatility forecasting provides the quantitative signal from which regime-transition risk can be derived.

Fallback:

- The older high-stress / low-stress classifier remains a useful emergency proxy, but is too crude to be the main Phase 2 problem framing.

## Target signal properties to emphasize

Volatility-regime transitions are attractive for reservoir computing because they combine:

- long memory and volatility clustering;
- multi-scale structure: short-, medium-, and longer-horizon volatility all matter;
- nonlinear coupling among realized volatility, VIX, returns, drawdowns, ranges, and volume;
- non-Gaussian shocks and heavy tails;
- structural instability and regime switching;
- weak but economically valuable predictability.

The submission should avoid implying that the problem is easy. The challenge statement explicitly treats regime-transition detection as hard, and this supports the use of a richer temporal feature map.

## QRC property to exploit

Weak justification:

- Quantum reservoirs have exponentially large Hilbert spaces, therefore they should help.

Stronger justification:

- A spin-system QRC provides tunable nonlinear fading-memory dynamics. The reservoir maps a compact multivariate market-history sequence into a high-dimensional observable feature vector. If the Hamiltonian/evolution time are chosen well, measured observables can expose transition-relevant nonlinear temporal features to a simple classical readout.

Properties to discuss:

- Hilbert-space dimensionality gives a large feature space, but this alone is not enough.
- Unitary/non-equilibrium dynamics provide nonlinear mixing of encoded inputs.
- Fading memory depends on evolution time, coupling strength, input injection, reset/re-uploading, and measurement strategy.
- Dissipation/noise may be harmful or regularizing; this is a Phase 3 study axis.
- Linear or ridge readout is useful because it tests whether the reservoir made the relevant signal more linearly accessible.

## Architecture-design checklist

Each proposed QRC architecture should specify:

- reservoir Hamiltonian or unitary structure;
- Hamiltonian parameters and evolution time;
- input features and scaling;
- encoding scheme: angle, phase, amplitude, direct feature-to-qubit, or re-uploading;
- encoding density: number of features per qubit / number of uploads;
- memory mechanism: rolling window, virtual nodes, recurrence, reset policy, feedback if used;
- readout observables: e.g., Z expectations, multi-Pauli observables, exact expectations vs shot estimates;
- classical head: ridge, linear regression, logistic/transition head, kernelized or polynomial readout if justified;
- resource estimate: qubits, circuit depth/evolution steps, shots, runtime;
- qBraid execution path: local simulator first, qBraid simulator/hardware later.

Likely first architecture:

- Reservoir: TFIM/spin-system QRC.
- Input: PCA-6 or selected volatility/VIX/drawdown/range features.
- Encoding: angle encoding with possible repeated encoding / virtual nodes.
- Memory: rolling temporal window injected sequentially; no mid-circuit feedback initially.
- Readout: Z or Pauli expectation features into ridge regression / transition-risk head.
- Output: volatility trajectory and derived transition-risk score.

## Lightweight prototype probes for Phase 2

The challenge does not require full final benchmarking in Phase 2. Prototype tests should substantiate design claims.

Useful probes:

1. Memory probe
   - Does the QRC state retain useful information over 5-20 trading-day horizons?
   - Vary evolution time or re-uploading depth.

2. Nonlinearity probe
   - Does QRC improve over a linear/HAR-like baseline on transition-relevant features or volatility trajectory prediction?
   - The goal is not final superiority; it is evidence of nontrivial signal extraction.

3. Encoding probe
   - Compare PCA-6 vs selected interpretable features.
   - Compare direct feature-to-qubit encoding vs fewer qubits with re-uploading.

4. Reservoir-dynamics probe
   - Vary coupling strength / field strength / evolution time.
   - Check whether dynamics are too weak, too chaotic/mixing, or useful.

5. Shot/noise feasibility probe
   - Compare exact expectations to small shot budgets such as 512/2048.
   - Optional simple depolarizing/amplitude damping pilot if time permits.

6. Regime-transition probe
   - Convert predicted volatility path into level/slope/persistence features.
   - Show at least one interpretable regime-transition warning example or event-level analysis.

## Phase 2 vs Phase 3 boundary

Phase 2 should answer:

- Is this a rigorous, well-justified, executable QRC design worth scaling in Phase 3?

Phase 2 should not try to finish:

- exhaustive hyperparameter optimization;
- complete scaling studies;
- full noise characterization;
- full LSTM/GARCH implementation unless needed for the narrative;
- final benchmark polishing.

## Submission narrative skeleton

1. Problem structure:
   - volatility-regime transitions are nonlinear, multi-scale, weakly predictable, and economically important.

2. QRC match:
   - TFIM/spin reservoirs provide nonlinear fading-memory feature maps suitable for compact multivariate temporal signals.

3. Design choice:
   - compact public-market features are encoded into a small spin reservoir; measured observables feed a classical readout.

4. Evidence:
   - small 7-12 qubit simulator prototypes demonstrate end-to-end execution and nontrivial predictive/feature signal.

5. Gap and Phase 3 plan:
   - full resource scaling, noise/shot studies, and stronger classical baselines are deferred to Phase 3.

## Rubric linkage

This section mainly supports:

- QRC Architecture Design;
- Theoretical & Analytical Justification;
- Track Selection & Problem Framing;
- Platform Justification & Resources;
- Phase 3 Execution Plan.
