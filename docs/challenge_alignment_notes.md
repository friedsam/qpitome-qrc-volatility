# Challenge alignment notes

Date: May 20, 2026

Status: living working notes. These notes capture current strategic decisions while reading the Phase 2 challenge description and QRC literature. They should be revised as the challenge brief is scanned further.

## Challenge framing

Track A is financial volatility prediction. The challenge asks for a QRC system that uses public equity-market data to identify volatility regime shifts and forecast their transitions.

The strongest alignment is therefore not a binary high-stress classifier alone. The primary benchmark should be volatility forecasting, with regime/transition interpretation derived afterward.

## Scope correction from current dev branch

The current `dev` branch contains a useful but partially misaligned emergency path:

- input: SPY/VIX-derived rolling market features;
- target: `future_high_vol_label`;
- task: classify high-stress vs non-stress future windows;
- metrics: PR-AUC, ROC-AUC, precision, recall, F1, balanced accuracy.

This is a simplified future-regime/stress proxy. It should be preserved as a fallback, but not treated as the strongest challenge-aligned primary task.

The alignment branch should instead emphasize:

- continuous realized-volatility forecasting;
- multi-horizon future volatility prediction;
- derived stress/regime/transition warnings from the predicted volatility path;
- challenge metrics: RMSE, QLIKE, Mincer-Zarnowitz;
- QRC design choices justified through signal class, Hamiltonian, encoding, noise, and baseline comparison.

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

## Dataset strategy

Challenge-suggested data sources include Yahoo Finance/Kaggle OHLCV, Oxford-Man realized-volatility data, and FRED macroeconomic time series.

Current data status:

- Yahoo-style SPY OHLCV + VIX-derived features: already used and useful.
- Oxford-Man realized-volatility library: not yet used; likely better aligned for realized-volatility forecasting.
- FRED macro data: optional later extension; do not add first because frequency alignment and macro lag structure add complexity.

Recommended target branch design:

- primary target: Oxford-Man realized volatility / realized variance if feasible;
- fallback target: current derived `future_rv_5d` / `future_rv_20d` from SPY/VIX pipeline;
- features: lagged realized-volatility features, SPY OHLCV features, VIX level/change, drawdown/range features;
- optional later: macro/FRED regime context only if the core pipeline is already working.

## Regime interpretation

A regime is not simply one thresholded volatility point. Better definition:

- persistent latent or observable market state;
- cluster/region in multidimensional temporal feature space;
- characterized by volatility level, volatility trend, VIX level/change, returns, ranges, drawdown, volume, and persistence.

Operational interpretation:

- primary model forecasts the volatility path;
- secondary logic derives regimes from forecast geometry:
  - calm: low predicted volatility, flat path;
  - unstable transition: moderate volatility with rising slope/acceleration;
  - stress/crisis: high and persistent predicted volatility.

Do not start by inventing mushy regime labels. First forecast volatility; then derive or test regime labels once forecast outputs exist. If scalar volatility forecasting underwhelms, upgrade early to multivariate trajectory/regime targets rather than over-tuning a bad target.

## QRC design philosophy

The challenge is asking for an experimental research program, not just a model. Every experiment should explicitly address one or more of:

1. signal class: scalar volatility, multivariate volatility trajectory, transition/regime dynamics, MNIST;
2. reservoir Hamiltonian: TFIM, Rydberg/Ising, random-unitary/RF-QRC;
3. encoding strategy: direct feature-to-qubit encoding, data re-uploading, virtual nodes/time multiplexing, amplitude/feature-map encoding;
4. noise/expressivity/baseline interaction: whether noise destroys, regularizes, or can be mitigated in reservoir states;
5. classical competition: HAR, ridge/persistence, ESN, LSTM, GARCH-family models where feasible.

Pre-run note template:

- Signal class:
- Hamiltonian:
- Encoding:
- Readout:
- Noise/shot setting:
- Classical baseline:
- Metric:
- Hypothesis:
- Decision rule:

Post-run note template:

- Result:
- Passed/failed decision rule:
- Likely bottleneck:
- Next action:
- Final-report relevance:

Time-control rule:

- Every experiment must answer one challenge-question axis.
- No experiment gets expanded unless it changes target choice, hardware choice, encoding choice, or final comparison.
- Do not reopen broad classical exploration unless needed for QRC comparison.

## Phase 2 rubric checkpoint

Use this 2-minute checkpoint after every completed milestone. Score each criterion qualitatively: strong / adequate / weak / missing.

1. QRC Architecture Design
   - Did this milestone clarify Hamiltonian, encoding, readout, feedback/no-feedback, or hybrid integration?
   - Evidence to capture: architecture diagram, equations, code module, or design note.

2. Theoretical & Analytical Justification
   - Did this milestone strengthen the case that QRC is appropriate for this signal class?
   - Evidence to capture: literature link, memory/nonlinearity argument, scaling/expressivity argument, or prototype result.

3. Data Modeling Strategy
   - Did this milestone improve dataset choice, preprocessing, target definition, baseline quality, or metrics?
   - Evidence to capture: data source, split logic, leakage checks, RMSE/QLIKE/Mincer-Zarnowitz, baseline table.

4. Track Selection & Problem Framing
   - Did this milestone make the Track A sub-problem clearer and more challenge-aligned?
   - Evidence to capture: volatility forecasting target, derived regime-transition interpretation, stakeholder-use framing.

5. Platform Justification & Resources
   - Did this milestone clarify backend choice, qubit count, depth/evolution time, shot budget, simulator/QPU path, or qBraid integration?
   - Evidence to capture: resource table, backend abstraction, runtime estimate, hardware plan.

6. Phase 3 Execution Plan
   - Did this milestone make the next-phase plan more concrete and feasible?
   - Evidence to capture: milestone timeline, fallback path, minimal viable final deliverable, risks.

7. Clarity of Communication
   - Did this milestone produce a clean figure/table/explanation usable in the submission?
   - Evidence to capture: report-ready plot, one-paragraph interpretation, README/runbook update.

Rule: if a milestone does not improve at least one rubric criterion, it probably should not be expanded.

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

## Phase 2 desired outcomes mapped to current work

Outcome 1: QRC architecture with explicit theoretical/analytical justification.

- Phase 2 relevance: required/central.
- Need now: Hamiltonian, input encoding, readout, feedback/no-feedback, and why these match volatility signals.

Outcome 2: Benchmark against strong classical baselines.

- Phase 2 relevance: required/central.
- Need now: persistence/HAR-like/ridge plus ESN; LSTM/GARCH can be optional, delegated, or included as future work if time is tight.

Outcome 3: Scaling with reservoir size, encoding density, shot budget, and noise.

- Phase 2 relevance: partial but important.
- Need now: pilot sweeps, not exhaustive final characterization.
- Minimal axes: 4/6/8 qubits or feasible sizes; direct vs re-uploading encoding; exact vs 512/2048 shots; noiseless vs simple depolarizing/amplitude damping.

Outcome 4: Fully reproducible qBraid workflow.

- Phase 2 relevance: not the full final requirement yet.
- Need now: qBraid-ready local workflow, clean dependencies, deterministic notebooks, saved outputs, backend abstraction.
- Actual qBraid credentials/cloud execution can wait; reproducibility debt should not accumulate.

## Revised experimental sequence

Experiment 0: target/data realignment.

- Build continuous realized-volatility targets.
- Prefer Oxford-Man target if feasible; otherwise current `future_rv_5d` / `future_rv_20d`.
- Implement RMSE, QLIKE, Mincer-Zarnowitz.

Experiment 1: classical forecasting baselines.

- Persistence.
- HAR-like ridge.
- Compact ridge / random forest / gradient boosting if fast.
- ESN regression.
- Goal: establish reproducible RMSE/QLIKE benchmark, not optimize classifier PR-AUC.

Experiment 2: first TFIM-QRC regression.

- Input: PCA-6 or selected compact volatility-market features.
- Reservoir: small TFIM QRC.
- Readout: ridge regression.
- Metrics: RMSE, QLIKE, Mincer-Zarnowitz.
- Decision: does QRC match or approach HAR/ESN?

Experiment 3: encoding-density sweep.

- Direct PCA-6 to 6 qubits.
- Fewer qubits with feature re-uploading.
- Selected interpretable physical features vs PCA features.

Experiment 4: reservoir-size/time/noise/shot pilot.

- Qubits: small feasible set.
- Evolution time: low/medium/high.
- Shots: exact, 512, 2048.
- Noise: none, depolarizing, amplitude damping.

Experiment 5: regime interpretation.

- Convert predicted volatility path to level/slope/persistence features.
- Evaluate derived calm/rising-instability/stress warnings.
- Only expand if regression outputs are meaningful or if scalar regression underwhelms and a multivariate regime target is the better QRC opportunity.

## qBraid readiness without credentials

Start qBraid-compatible reproducibility now, but do not block on credentials.

Do now:

- environment files;
- backend abstraction: `local_simulator`, later `qbraid_simulator`, later optional `qbraid_qpu`;
- notebooks runnable top-to-bottom locally;
- no hidden paths or local-only credentials;
- saved tables/figures under `results/`;
- qBraid runbook draft.

Wait for credentials for:

- actual qBraid cloud jobs;
- backend discovery;
- hardware/backend execution validation;
- judge-style rerun test.
