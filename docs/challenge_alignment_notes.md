# Challenge alignment notes

Date: May 20, 2026

## Challenge framing

Track A is financial volatility prediction. The challenge asks for a QRC system that uses public equity-market data to identify volatility regime shifts and forecast their transitions.

The strongest alignment is therefore not a binary high-stress classifier alone. The primary benchmark should be volatility forecasting, with regime/transition interpretation derived afterward.

## Recommended Track A metrics from the challenge brief

- RMSE: standard regression error.
- QLIKE: volatility-specific quasi-likelihood loss.
- Mincer-Zarnowitz regression: realized values regressed on forecasts; test intercept = 0 and slope = 1 for forecast unbiasedness/efficiency.

Classification metrics such as PR-AUC, ROC-AUC, F1, precision, and recall are useful secondary diagnostics for derived stress or regime-warning labels, but they are not the primary Track A metrics.

## Literature ballpark from Li et al. 2025 QRC realized-volatility paper

For S&P 500 realized-volatility forecasting:

### One-step ahead, S = 1

- QR2: MSE about 0.103, QLIKE about 1.4004.
- QR1: MSE about 0.105, QLIKE about 1.4427.
- Best classical RCX: MSE about 0.1089, QLIKE about 1.6480.
- LSTM: MSE about 0.1295.
- HAR: MSE about 0.1476.

### Five-step ahead, S = 5

- Best classical RC: MSE about 0.1528, QLIKE about 2.0551.
- QR1: MSE about 0.1556.
- QR2: MSE about 0.1663.
- LSTM: MSE about 0.1831.
- HAR: MSE about 0.2143.

Interpretation: expected QRC advantage is modest, not dramatic. Matching the strongest classical reservoir within error bars is already useful. A small improvement in RMSE/QLIKE, better robustness, or better scaling/noise behavior would be meaningful.

## Project implication

Current branch preserves the emergency/high-stress-label approach. The next alignment branch should pivot to:

1. continuous volatility forecasting: future_rv_5d / future_rv_20d or multi-horizon volatility vector;
2. Track A metrics: RMSE, QLIKE, Mincer-Zarnowitz;
3. derived regime/transition metrics as secondary interpretation;
4. QRC architecture justification tied to nonlinear multivariate temporal structure, memory, and reservoir expressivity;
5. qubit-count, encoding-density, shot-budget, and noise studies for Phase 3 planning.

## QRC hardware/reservoir-system primers

The challenge description lists several possible quantum reservoirs: transverse-field Ising chains, Rydberg atom arrays, and cavity QED systems. These are not just circuit templates; they correspond to different physical hardware strategies and different implementation risks.

### Transverse-field Ising chain

Hardware mapping:

- gate-based superconducting or trapped-ion platforms through digital simulation;
- neutral-atom analog systems when formulated as Ising/Rydberg dynamics;
- classical statevector/density-matrix simulators for first implementation.

Pros:

- strongest default choice for this project;
- closest to the QRC volatility-forecasting literature;
- easy to simulate first;
- clean Hamiltonian story;
- compatible with small qubit-count sweeps;
- can be approximated with gate-based circuits if analog access is unavailable.

Cons:

- gate-based depth/noise can limit usable memory;
- analog interactions/geometries may not match the ideal model exactly;
- fully connected or dense Ising dynamics are easier in simulation than on hardware.

Strategy implication:

- Primary implementation path: simulator-based transverse-field Ising QRC for volatility forecasting.
- Use as the baseline QRC architecture before trying more hardware-specific variants.

### Rydberg atom array

Hardware mapping:

- neutral-atom analog platforms such as QuEra/Aquila-style systems;
- Bloqade-style simulation and prototyping.

Pros:

- strong conceptual fit to analog QRC and scalable many-body dynamics;
- good story for hardware-informed reservoir computing;
- natural connection to Rydberg/Ising dynamics and large Hilbert spaces;
- potential Phase 3 hardware/scaling narrative.

Cons:

- input encoding is less flexible than generic gate-based circuits;
- geometry/connectivity constraints matter;
- hardware access and queue/runtime constraints may limit iteration;
- measurement/readout options may be less flexible.

Strategy implication:

- Strong hardware-aligned extension after the simulator pipeline works.
- Not the first debugging target unless the challenge/rubric strongly rewards hardware-specific analog design.

### Cavity QED / few-body feedback reservoirs

Hardware mapping:

- specialized photonic or cavity-QED experimental systems;
- generally not a standard qBraid-accessible backend.

Pros:

- strong memory/feedback and physical-reservoir story;
- useful conceptual support for minimal QRC and rich dynamics from few degrees of freedom.

Cons:

- likely not implementable in our qBraid workflow;
- harder to reproduce;
- weaker near-term submission path.

Strategy implication:

- Cite as conceptual motivation only, not as implementation backbone.

### Gate-based random-unitary / RF-QRC fallback

Hardware mapping:

- IBM, IonQ, Rigetti, IQM, Braket/qBraid simulators, and other circuit-model platforms.

Pros:

- most portable;
- easy to reproduce;
- straightforward qubit-count, shot-budget, and noise sweeps;
- fits finite-sampling and noisy-QRC analysis.

Cons:

- less physically distinctive than analog QRC;
- may require many shots;
- advantage story can be weaker unless tied to robustness, compactness, or scaling.

Strategy implication:

- Best fallback/reproducibility path.
- Use as a comparison architecture if TFIM or Rydberg path becomes blocked.

## Current hardware-choice recommendation

Primary route:

- transverse-field Ising QRC on simulator;
- task: multivariate realized-volatility forecasting;
- metrics: RMSE, QLIKE, Mincer-Zarnowitz;
- secondary: derived regime/transition-warning metrics.

Hardware-aligned extension:

- Rydberg/neutral-atom Ising-style QRC via Bloqade/QuEra path.

Fallback:

- gate-based RF-QRC or random-unitary QRC with the same input/output/metrics.

Expected result target:

- do not promise large quantum advantage;
- aim to match strong classical reservoir baselines, show small RMSE/QLIKE gains if available, and document qubit-count/noise/shot-budget behavior;
- emphasize compact reservoir/readout design and alignment with nonlinear multivariate temporal dynamics.
