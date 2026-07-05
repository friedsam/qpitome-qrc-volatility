# Phase 2/Phase 3 challenge requirements and QRC design guidance

Last updated: 2026-07-05

## Purpose

This document is the permanent project checklist and design guide derived from the challenge materials. It has two jobs:

1. prevent us from missing a required Phase 3 deliverable, benchmark, metric, scaling study, or reproducibility item;
2. preserve the challenge organizers' actual QRC design hints so earlier pipeline decisions can be judged against the intended problem.

This is not a summary of our current model results. It is the external specification against which the project should be audited.

---

## 1. Source hierarchy and known document problems

Sources reviewed:

1. Phase 2 detailed challenge PDF;
2. Phase 3 challenge PDF;
3. Phase 3 webpage text captured in `Quantum Reservoir Computing for Time.docx`.

### Authority rule

Use the **Phase 3 webpage text as the authoritative Phase 3 execution specification** when it conflicts with obvious errors in the Phase 3 PDF.

The Phase 3 PDF contains contamination from another challenge, including a requirement to run three pipeline stages on QCi Dirac-3 for a concrete grid instance. That requirement is unrelated to Dynamic Systems Forecasting and must be ignored.

The Phase 2 PDF remains authoritative for the detailed Track A problem framing, QRC architecture requirements, theory requirements, data-modeling expectations, recommended baselines, metrics, platform planning, and stakeholder framing unless Phase 3 explicitly updates them.

### Practical implication

Before final submission, re-check the live Aqora page for page limits, cover-page rules, AI-use rules, and upload instructions. Do not rely only on the buggy Phase 3 PDF.

---

# PART A — COMPLIANCE AND EXECUTION CHECKLIST

## 2. Track and problem definition

### Required

- [x] Select Track A: Financial Volatility Prediction.
- [ ] State the exact forecasting subproblem in one sentence.
- [ ] State the forecast target mathematically.
- [ ] State the temporal aggregation and forecast horizon.
- [ ] State why the selected task addresses volatility regime shifts and transition forecasting.
- [ ] State why the task is operationally useful to stakeholders.
- [ ] Use publicly available equity-market data only.
- [ ] Document exact source, frequency, time range, and access method for every dataset.

### Challenge wording that matters

The challenge asks for a QRC system that can:

- identify volatility regime shifts;
- forecast their transitions;
- use publicly available equity-market data.

It explicitly allows teams to choose the subproblem, with realized-variance forecasting at a chosen horizon given as an example.

### Project implication

The primary output can remain a continuous volatility forecast. Regime-transition value should be demonstrated through target choice, evaluation slices, or auxiliary outputs rather than by forcing the project into crisis classification.

---

## 3. Stakeholder relevance

### Required stakeholder story

Track A stakeholders named or implied by the challenge include:

- trading desks;
- risk managers;
- market makers;
- portfolio managers;
- derivatives and volatility desks.

### Questions the final project should answer

- [ ] What decision changes if the forecast improves?
- [ ] Does the method provide earlier warning of increasing risk?
- [ ] Does it distinguish onset, persistence, and decay of elevated volatility?
- [ ] Does it improve risk pricing or position-sizing inputs?
- [ ] Does it reveal higher-order state interactions that are not visible to standard models?
- [ ] Is the gain large enough to matter relative to runtime, complexity, and hardware cost?

### Important correction to earlier project drift

A crisis-only project is too narrow. The challenge asks about regime shifts broadly, not only market crashes. Stakeholder value can include:

- transition warning;
- persistence of elevated-risk regimes;
- de-escalation/recovery detection;
- ongoing risk-state monitoring;
- forecast calibration under changing market conditions.

---

## 4. Required QRC architecture specification

The challenge treats architecture design as the technical core.

The final project must specify and justify all of the following.

### 4.1 Reservoir dynamics

- [ ] Hamiltonian or unitary structure.
- [ ] Why that structure creates useful temporal dynamics.
- [ ] Topology/connectivity.
- [ ] Reservoir size in qubits/atoms.
- [ ] Evolution duration or circuit depth.
- [ ] Reset/state-carrying rule between timesteps.

Challenge examples:

- transverse-field Ising;
- Rydberg analog evolution;
- gate-based ansatz with fixed random parameters;
- Ising;
- Heisenberg;
- random graphs;
- Bose-Hubbard;
- custom topologies.

### 4.2 Input encoding

- [ ] Exact mapping from classical variables to control parameters.
- [ ] Scaling domain and clipping rule.
- [ ] Encoding density: how many classical variables per qubit/atom/control channel.
- [ ] Whether inputs are encoded once or repeatedly during evolution.
- [ ] Whether temporal multiplexing is used.

Challenge examples:

- amplitude encoding;
- angle encoding;
- phase encoding;
- repeated encoding;
- temporal multiplexing.

### 4.3 Measurement/readout

- [ ] Which observables are measured.
- [ ] Whether readout uses exact expectation values, finite-shot estimates, or hardware samples.
- [ ] Whether spatial or temporal multiplexing creates additional virtual features.
- [ ] Exact classical readout model and regularization.

Challenge examples:

- projective measurement;
- weak measurement;
- continuous measurement;
- expectation values;
- shot-based estimators;
- spatial multiplexing across parallel reservoirs;
- ridge regression;
- polynomial regression;
- kernelized or expanded classical readouts.

### 4.4 Memory and feedback

- [ ] Explain where fading memory comes from.
- [ ] State whether quantum state is carried between inputs.
- [ ] State whether measurement outcomes are re-injected.
- [ ] State whether a classical memory layer is used.

Challenge examples:

- recurrent feeding of measurements;
- mid-circuit measurement;
- classical memory qubits;
- measurement re-injection to restore fading memory and enhance nonlinearity;
- classical memory augmentation around quantum feature extraction.

### 4.5 Hybrid integration

- [ ] Define the boundary between classical preprocessing, quantum evolution, and classical readout.
- [ ] Explain why each operation belongs on that side of the boundary.
- [ ] Avoid hiding the predictive task almost entirely in classical feature engineering.

---

## 5. Theoretical and analytical justification

This section is heavily weighted.

The final argument must identify:

### 5.1 Quantum property being exploited

At least one specific mechanism must be named and tested:

- [ ] high-dimensional Hilbert-space feature expansion;
- [ ] intrinsic nonlinearity of quantum dynamics/measurement map;
- [ ] memory from non-equilibrium dynamics;
- [ ] interference/correlation structure;
- [ ] dissipation/noise as a useful feature or regularizer;
- [ ] higher-order observable structure;
- [ ] feedback-enhanced fading memory.

### 5.2 Matching signal property

The mechanism must be connected to an empirically demonstrated property of the target signal:

- [ ] long memory;
- [ ] multiscale behavior;
- [ ] regime switching;
- [ ] non-Gaussian behavior;
- [ ] nonlinear state interactions;
- [ ] chaotic dynamics, if a defensible chaos measure exists.

### 5.3 Evidence requirement

- [ ] Show a small-scale experiment supporting the claimed mechanism.
- [ ] Include a control or ablation that could falsify the mechanism.
- [ ] Distinguish performance evidence from mechanistic evidence.

### Project rule

Do not write a generic argument that “quantum is nonlinear and finance is nonlinear.” The challenge explicitly asks for a matched mechanism-to-signal justification.

---

## 6. Data-modeling requirements

### Required documentation

- [x] Public data provenance.
- [x] Frequency and time range.
- [x] Missingness protocol.
- [x] Leakage-aware feature timing.
- [ ] Final target definition.
- [ ] Final train/validation/test protocol.
- [ ] Final normalization and fitting boundaries.
- [ ] Final windowing/state-carrying rule.
- [ ] Identical evaluation dates across all benchmark models.

### Classical baselines explicitly requested for Track A

- [ ] GARCH.
- [x] ESN implementation exists.
- [ ] LSTM.

Additional diagnostics allowed and useful:

- [x] persistence;
- [x] AR(1);
- [x] HAR;
- [x] HARX;
- [x] Ridge.

### Critical challenge statement

The ESN is described as the **most important baseline**, because beating the classical analog of QRC is what would justify QRC.

### Project implication

A QRC result that only beats persistence, AR, or HAR is insufficient if ESN remains stronger.

---

## 7. Required Track A metrics

### Core metrics

- [x] RMSE implementation.
- [x] QLIKE implementation.
- [x] Mincer-Zarnowitz regression implementation.

### Final reporting requirements

- [ ] Use identical target units and transformations for all models.
- [ ] State whether RMSE is on RV, variance, or log-RV.
- [ ] State the exact QLIKE formula and whether it is evaluated on variance or another transformed target.
- [ ] Report MZ intercept and slope.
- [ ] Report uncertainty or variation across stochastic seeds where applicable.
- [ ] Report regime/transition-specific performance in addition to global metrics if used in the project claim.

---

## 8. Qubit scaling requirements

Phase 3 asks for simulation-friendly QRC systems in approximately the 5–20 qubit range and performance across different qubit counts.

### Required scaling study

- [ ] At least three reservoir sizes.
- [ ] Include representative values such as 5, 10, 15 qubits/atoms where feasible.
- [ ] Hold the problem instance and classical evaluation dates fixed.
- [ ] Report predictive metrics versus reservoir size.
- [ ] Report runtime versus reservoir size.
- [ ] Report feature/readout dimension versus reservoir size.
- [ ] Explain where simulation or hardware limits appear.

### Project implication

The Rydberg plan should not jump directly to one reservoir size. Architecture code must expose reservoir size as a first-class configuration.

---

## 9. Encoding-density and memory-capacity study

Phase 2 explicitly asks how performance scales with encoding density. Phase 3 design hints also emphasize temporal multiplexing and feedback.

### Required or strongly indicated

- [ ] Compare at least two input-density strategies.
- [ ] Compare compact versus overloaded input maps.
- [ ] Quantify the tradeoff between input information and reservoir memory capacity.
- [ ] If using temporal multiplexing, vary the number of virtual nodes/readout times.
- [ ] If using repeated encoding, vary repetition count or injection schedule.

### Direct relevance to our pipeline

The paper sanity study already showed:

- compact seven-feature linear models retained nearly all performance of 24 features;
- a 200-node ESN overfit badly while a 50-node ESN recovered;
- more input/readout capacity was not automatically better.

This supports treating encoding density and capacity allocation as scientific variables, not implementation details.

---

## 10. Shot-budget study

The challenge explicitly asks how performance scales with shot budget.

### Required

- [ ] Exact/noiseless expectation-value reference where possible.
- [ ] At least several finite-shot budgets.
- [ ] Report predictive metric degradation versus shots.
- [ ] Report ranking stability or feature-geometry stability where appropriate.
- [ ] Report wall-clock or sampling cost.
- [ ] Use the same trained readout protocol across fair comparisons.

### Project implication

The desired evidence chain remains:

```text
exact simulator -> finite shots -> hardware-native execution -> Aquila
```

A claimed quantum advantage that disappears under realistic sampling is not a useful Phase 3 result.

---

## 11. Noise study

The Phase 3 webpage explicitly asks for realistic noise models, including examples of depolarizing noise and amplitude damping.

### Required

- [ ] Define the noise model used for simulator studies.
- [ ] Sweep at least several noise strengths.
- [ ] Report predictive performance degradation.
- [ ] Report whether noise changes feature geometry or ranking.
- [ ] Distinguish gate-model noise assumptions from analog-hardware noise.

### For Rydberg/Aquila

Do not blindly apply gate-noise channels to an analog system and call that realistic.

Use two layers:

1. challenge-compatible generic noise sensitivity where useful;
2. hardware-native imperfections/noise model appropriate to Rydberg evolution and Aquila execution.

The challenge materials also cite prior work where hardware noise may act as regularization. This suggests an allowed but testable hypothesis:

> moderate noise may reduce overfitting or improve generalization in some small-data QRC regimes.

That must be demonstrated, not assumed.

---

## 12. Hardware and platform requirements

### Required planning/reporting

- [ ] Simulator backend(s).
- [ ] Hardware backend(s).
- [ ] Qubit/atom count.
- [ ] Circuit depth or analog evolution duration.
- [ ] Simulation settings.
- [ ] Shot budget.
- [ ] Wall-clock runtime.
- [ ] Classical compute integration.
- [ ] Fallback when simulator or QPU limits are hit.

### Challenge guidance

Simulator-first development is encouraged. QPU access should be used for targeted scaling and validation rather than wasting scarce credits on uncontrolled sweeps.

### Project plan

- TFIM remains the Phase 2-equivalent control/reference.
- Rydberg is the primary Phase 3 QRC stream.
- Aquila should be used early enough to expose hardware constraints, but only after a defensible encoding exists.

---

## 13. Classical-versus-quantum comparison rules

### Required

- [ ] Same problem instance.
- [ ] Same target definition.
- [ ] Same forecast dates.
- [ ] Same train/test information boundary.
- [ ] Comparable feature availability.
- [ ] Explicit computational-resource reporting.

### Quantum advantage must be concrete

Allowed forms of benefit include:

- predictive improvement;
- extraction of higher-order patterns;
- new insight difficult to obtain classically;
- computational benefit;
- robustness/noise benefit.

But the final writeup must show numbers. Qualitative claims alone are explicitly insufficient.

### Project standard

A headline claim should survive:

1. strong classical baseline;
2. multiple seeds where stochasticity matters;
3. finite shots;
4. realistic noise;
5. at least targeted hardware validation;
6. transparent runtime and resource accounting.

---

## 14. Common MNIST QRC benchmark

The Phase 3 webpage adds a common MNIST digit-classification benchmark for all teams.

Purpose:

- standardized comparison across teams;
- check that the QRC architecture has sufficient expressivity.

### Required

- [ ] Implement MNIST through the same core QRC architecture family.
- [ ] Define data reduction/encoding clearly.
- [ ] Report accuracy and resource usage.
- [ ] Compare across qubit counts.
- [ ] Include noise sensitivity.
- [ ] Keep MNIST code separate from the financial pipeline.

### Project implication

MNIST is an architecture-expressivity benchmark, not a reason to alter the financial target or feature pipeline.

---

## 15. Reproducibility and qBraid execution

Phase 3 strongly emphasizes judge reruns.

### Submission package

- [ ] Zip filename: `TeamName_Challenge_Phase3.zip`.
- [ ] Maximum 5-page writeup PDF, references excluded.
- [ ] 11-point Times New Roman.
- [ ] Single spacing.
- [ ] Required cover-page template verified from live Aqora instructions.
- [ ] Source code folder included.
- [ ] Code organized by module.
- [ ] Code executable on qBraid without external configuration.
- [ ] `README.md` included.

### README must include

- [ ] Team name.
- [ ] Project title.
- [ ] Challenge track.
- [ ] Setup instructions.
- [ ] Dependencies/packages.
- [ ] Step-by-step qBraid execution instructions.
- [ ] Expected inputs.
- [ ] Expected outputs.
- [ ] Known limitations and assumptions.
- [ ] Launch on qBraid button.

### Reproducibility test

- [ ] Fresh-environment run.
- [ ] Judge-like run without manual edits.
- [ ] Headline results reproduced.
- [ ] Seeds/configs stored.
- [ ] Raw data acquisition documented.
- [ ] Cached hardware results clearly distinguished from reproducible simulator runs.

---

## 16. qBraid Skill requirement

The Phase 3 webpage asks teams to submit a qBraid Skill: an agent-executable package that can navigate the codebase, configure the reservoir, train, and reproduce results end-to-end.

### Required functional capabilities

- [ ] Select dataset/task.
- [ ] Select reservoir architecture.
- [ ] Configure reservoir size.
- [ ] Configure encoding.
- [ ] Configure shot/noise settings.
- [ ] Run training/readout fitting.
- [ ] Run evaluation.
- [ ] Produce a standard result artifact.

### Project implication

The repository must remain configuration-driven and modular. Hard-coded notebook sequences or chat-dependent manual steps will fail this requirement.

---

## 17. Final writeup numbers that must appear

The Phase 3 webpage explicitly asks for concrete numbers.

At minimum:

- [ ] qubit/atom count;
- [ ] circuit depth or analog evolution duration;
- [ ] shot budget;
- [ ] wall-clock runtime;
- [ ] RMSE;
- [ ] QLIKE;
- [ ] MZ calibration;
- [ ] classical baseline values;
- [ ] scaling results;
- [ ] noise results;
- [ ] hardware result if used;
- [ ] MNIST result;
- [ ] limitations.

---

## 18. Honesty and limitations

The challenge explicitly rewards honest understanding of where the quantum approach does and does not help.

### Required

- [ ] State whether the advantage is predictive, mechanistic, computational, or absent.
- [ ] State where classical methods remain better.
- [ ] State sensitivity to seed, shots, noise, and reservoir size.
- [ ] State simulation/hardware gaps.
- [ ] Avoid claiming general quantum advantage from one favorable split or one seed.

---

# PART B — WHAT THE CHALLENGE WANTS US TO TRY FOR QRC

## 19. The intended scientific question

The challenge is not merely asking for a quantum version of an ordinary forecasting pipeline.

Its central open questions are:

- which Hamiltonians are matched to which signal classes;
- which encoding strategies are matched to temporal structure;
- how memory and nonlinearity should be balanced;
- how noise changes expressivity;
- where QRC genuinely competes with strong classical reservoirs.

This should influence target and feature design.

---

## 20. QRC design hypotheses worth testing

### 20.1 Match reservoir timescale to signal timescale

Challenge signal properties include long memory and multiscale dynamics.

Implication:

- estimate empirically which temporal aggregation remains linearly explainable by HAR;
- identify scales where persistence remains but linear multiscale approximation weakens;
- tune reservoir leak/evolution duration/injection schedule to those scales.

This motivates the proposed temporal-scale map:

```text
daily aggregation -> next-scale forecast
weekly aggregation -> next-scale forecast
monthly aggregation -> next-scale forecast
```

without overlapping forward labels.

HAR acts as a linear multiscale reference, not as a target-selection trick.

### 20.2 Compact encoding versus reservoir memory

The challenge explicitly highlights encoding density and memory.

Implication:

- do not blindly feed all engineered features into the quantum reservoir;
- compare compact physical feature groups against dense encodings;
- preserve hidden reservoir degrees of freedom for memory and nonlinear mixing.

Candidate compact groups:

- volatility state/memory;
- market shock/reversal;
- credit stress;
- macro dynamics.

### 20.3 Temporal multiplexing

The Phase 3 design space explicitly names temporal multiplexing.

Implication:

- sample observables at multiple evolution times after each input;
- treat intermediate readout times as virtual nodes;
- test whether this improves multiscale temporal representation without more qubits.

This is especially attractive for analog Rydberg QRC.

### 20.4 Repeated encoding

Repeated injection can couple new inputs to persistent reservoir state.

Implication:

- compare one-shot encoding with multiple injections during an evolution cycle;
- test whether repeated encoding improves transition forecasting or only increases redundancy.

### 20.5 Feedback/re-injection

The challenge explicitly suggests feeding measurements back to restore fading memory and enhance nonlinearity.

Implication:

- consider a lightweight classical feedback channel from previous measured observables to the next input encoding;
- compare against no-feedback control;
- use only if it preserves hardware feasibility and reproducibility.

### 20.6 Spatial multiplexing / parallel reservoirs

The challenge allows spatial multiplexing across parallel reservoirs.

Implication:

- small reservoirs with different parameters/topologies may provide more diverse features than one overloaded reservoir;
- compare one 15-qubit reservoir against multiple smaller reservoirs only if resource accounting remains fair.

### 20.7 Hybrid classical memory plus quantum feature extraction

The challenge explicitly allows classical memory augmentation.

Implication:

- QRC does not need to carry every long timescale internally;
- classical low-dimensional state summaries can provide slow memory while the quantum reservoir extracts nonlinear interactions;
- this may fit the finance task better than forcing a small quantum device to memorize 12 months directly.

A defensible architecture could therefore separate:

```text
slow classical state -> compact controls
recent temporal stream -> quantum reservoir dynamics
quantum observables -> linear/ridge readout
```

### 20.8 Noise as regularization

The challenge reference set and Phase 3 emphasis suggest testing whether hardware/noise can regularize small-data QRC.

Implication:

- compare noiseless, finite-shot, noisy-simulator, and hardware results;
- ask whether moderate noise improves test performance or feature stability;
- do not claim this from one hardware run.

### 20.9 Higher-order readout structure

The challenge allows polynomial and kernelized readouts.

Implication:

- first use linear/ridge readout to preserve the QRC premise;
- only add polynomial interactions as an explicit ablation;
- if a classical nonlinear readout creates the gain, attribute it honestly rather than to the quantum reservoir.

---

## 21. How this should influence earlier pipeline decisions

### Target selection

Do not select a target only because QRC wins on it.

Use a two-stage scientific screen:

1. stakeholder validity and challenge alignment;
2. dynamical structure analysis.

Then ask whether reservoir gain appears where linear multiscale models fail.

### Temporal aggregation

Do not automatically move to shorter RV horizons.

The paper uses monthly aggregation, which combines market, credit, and macro dynamics on compatible timescales. The old daily forward-window project was dominated by overlap and local persistence.

The next empirical check should map linear explainability across naturally aggregated scales rather than repeat 5/10/20-day overlapping targets.

### Feature engineering

Features should support three roles:

- state;
- dynamics;
- external drivers.

But the QRC input map should remain compact enough to preserve reservoir capacity.

### Baseline strategy

Use HAR as a linear multiscale diagnostic and ESN as the decisive reservoir baseline.

Potentially informative pattern:

```text
HAR strong, ESN similar -> little reservoir headroom
HAR weak, ESN strong -> reservoir-relevant task
HAR weak, ESN weak -> likely noise rather than useful nonlinearity
```

This is a hypothesis to test, not a selection rule.

### Hardware strategy

Design for the Phase 3 requirements now:

- configurable qubit count;
- finite-shot mode;
- noise mode;
- runtime logging;
- hardware-compatible encoding;
- reproducible simulator fallback.

---

## 22. Recommended active work order

### Immediate

1. [ ] Build temporal-scale EDA using non-overlapping natural aggregation.
2. [ ] Compare HAR explainability across scales.
3. [ ] Compare a controlled ESN across the same scales.
4. [ ] Decide the challenge target from stakeholder relevance plus dynamical evidence.

### Then

5. [ ] Freeze the challenge benchmark protocol.
6. [ ] Run GARCH, ESN, LSTM, and financial diagnostics.
7. [ ] Rebuild the Phase 2-equivalent TFIM control on the selected challenge task.
8. [ ] Implement Rydberg simulator architecture.
9. [ ] Test compact encoding density.
10. [ ] Test temporal multiplexing.
11. [ ] Test reservoir size scaling.
12. [ ] Test shot budgets.
13. [ ] Test realistic noise.
14. [ ] Run targeted Aquila validation.
15. [ ] Implement MNIST architecture benchmark.
16. [ ] Build qBraid Skill and judge-reproduction path.
17. [ ] Assemble final 5-page evidence-driven writeup.

---

## 23. Final project success criterion

A strong submission should be able to say, with evidence:

> We chose a stakeholder-relevant volatility-transition forecasting task because its temporal structure leaves measurable headroom beyond linear multiscale models. We matched that structure to a specific QRC mechanism, benchmarked against GARCH, ESN, and LSTM on identical data, demonstrated scaling across reservoir sizes, quantified shot and noise sensitivity, validated the architecture on hardware where feasible, and reproduced the result end-to-end on qBraid.

If the quantum model does not win, the project can still score well if it rigorously identifies where and why it fails, provides reproducible evidence, and shows a meaningful mechanistic insight.
