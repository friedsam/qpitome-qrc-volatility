# Phase 2 milestone plan: May 20-May 30

Date created: May 20, 2026
Branch: `phase2-volatility-regression-qrc`

May 31 is reserved for final paper writing and assembly. The goal of May 20-May 30 is to make the paper mostly mechanical to write: every milestone should leave behind a report-ready paragraph, table, figure, code artifact, or design note.

## Non-negotiable operating rules

1. Phase 2 is design-rigor first, not final benchmark dominance.
2. Regime-transition early warning is the headline Track A problem.
3. Realized-volatility / realized-variance forecasting is the required quantitative substrate.
4. RMSE, QLIKE, and Mincer-Zarnowitz are central Track A metrics, not optional or secondary.
5. QRC experiments should be light-touch prototypes that substantiate design claims.
6. No experiment should expand unless it improves at least one rubric criterion.
7. Simulator-first; qBraid/QPU execution must not block Phase 2.

## Required reference documents

Primary challenge source:

- `Qbraid MITRE JonesTrading_Phase 2 Challenge Description (1).pdf`

Project notes created during planning:

- `docs/challenge_alignment_notes.md`
- `docs/phase2_execution_guardrails.md`
- `docs/theoretical_justification_notes.md`
- `docs/track_a_metrics_notes.md`
- `docs/platform_stakeholder_phase3_notes.md`

Use these as sign-off checklists. Do not rely on memory.

## Daily milestone plan

### May 20 — Milestone 0: Freeze alignment and scope

Goal:

- Finalize the working Phase 2 interpretation and freeze scope boundaries.

Outputs:

- Updated notes committed to repo.
- This milestone plan committed to repo.
- One-paragraph current project framing:
  - Track A.
  - Regime-transition early warning.
  - Realized-volatility forecasting substrate.
  - TFIM/spin-QRC prototype.

Required checks before sign-off:

- Challenge description: problem statement, desired outcomes, Phase 2 vs Phase 3 boundary, rubric.
- `docs/challenge_alignment_notes.md`: challenge framing, scope correction, rubric checkpoint.
- `docs/phase2_execution_guardrails.md`: light-touch prototype boundary.
- `docs/track_a_metrics_notes.md`: metrics hierarchy.

Rubric criteria addressed:

- Track Selection & Problem Framing.
- Clarity of Communication.
- Phase 3 Execution Plan, initial boundary.

Stop condition:

- We can state the project in one precise paragraph without drifting into old binary-classification framing.

---

### May 21 — Milestone 1: Data modeling specification

Goal:

- Define exact dataset strategy and leakage-safe preprocessing pipeline.

Core decisions:

- Dataset source(s): Oxford-Man RV if feasible; fallback/current SPY/VIX/Yahoo-style data.
- Frequency and date range.
- Target: realized volatility / realized variance at selected horizon(s).
- Regime-transition definition: derived from realized-volatility trajectory, not an arbitrary standalone label.
- Features: interpretable compact features and/or PCA-6 version.
- Splits: chronological train/validation/test only.

Outputs:

- `docs/data_modeling_strategy.md` or equivalent section.
- Dataset/preprocessing module or notebook skeleton.
- Data dictionary: source, feature, target, lag, whether feature is known at prediction time.
- Leakage checklist.

Required checks before sign-off:

- Challenge description: Data Modeling Strategy, Track A datasets, Track A metrics.
- `docs/challenge_alignment_notes.md`: dataset strategy, regime interpretation, revised experimental sequence.
- `docs/track_a_metrics_notes.md`: required metrics and quantitative substrate.
- `docs/theoretical_justification_notes.md`: target signal properties.

Rubric criteria addressed:

- Data Modeling Strategy.
- Track Selection & Problem Framing.
- Theoretical & Analytical Justification.

Stop condition:

- A third party can understand exactly what data, target, windows, and splits will be used.

Scope-control warning:

- Do not add FRED unless the core public equity/RV pipeline is already clean.

---

### May 22 — Milestone 2: Implement preprocessing and required metrics

Goal:

- Build runnable, reproducible data pipeline and Track A metric functions.

Outputs:

- Code module for:
  - data loading/alignment;
  - normalization fit on train only;
  - rolling windows;
  - realized-volatility / variance targets;
  - transition labels/scores derived from volatility path;
  - chronological splits.
- Metric module for:
  - RMSE;
  - QLIKE;
  - Mincer-Zarnowitz regression.
- First-test notebook/table showing train/val/test sizes and no obvious leakage.

Required checks before sign-off:

- Challenge description: Data Modeling Strategy and Track A metrics.
- `docs/track_a_metrics_notes.md`.
- `docs/challenge_alignment_notes.md`: qBraid readiness and reproducibility expectations.
- `docs/phase2_execution_guardrails.md`: minimum useful prototype.

Rubric criteria addressed:

- Data Modeling Strategy.
- Clarity of Communication.
- Phase 3 Execution Plan, because reproducible data code reduces future risk.

Stop condition:

- Running the notebook/script produces a clean dataset object and metric outputs without manual path edits.

Scope-control warning:

- Do not optimize models yet. This milestone is data and metrics only.

---

### May 23 — Milestone 3: Classical baseline floor

Goal:

- Establish credible classical baselines sufficient for Phase 2 design justification.

Minimum baselines:

- Persistence / naive forecast.
- HAR-like ridge or linear model.
- ESN regression if feasible without large sweeps.

Optional / Phase 3 noted, not required now:

- GARCH-family.
- LSTM / GRU sequence models.

Outputs:

- Baseline notebook/script.
- Metrics table: RMSE, QLIKE, Mincer-Zarnowitz for each baseline.
- One paragraph interpreting baseline difficulty and what QRC must plausibly improve or match.

Completion artifacts:

- `docs/classical_baseline_notes.md`.
- `notebooks/phase2_esn_regression_baseline_tuning.ipynb`.
- `src/qpitome_qrc/baselines/esn_regression.py`.
- `src/qpitome_qrc/data/pca.py`.
- Curated ESN/PCA result tables if force-added despite `gitignore`.

Completion note:

- The classical floor is stronger than the original minimum plan. In addition to persistence and HAR-like/regularized linear regression, a ReservoirPy ESN regression baseline was adapted from the earlier binary-classification prototype to the current realized-volatility regression framing. The useful ESN version uses PCA-compressed volatility/VIX inputs, 40-day memory, ridge readout on log-volatility, and RMSE/QLIKE/Mincer-Zarnowitz evaluation. This establishes a meaningful classical reservoir comparator for QRC, not merely a naive or primitive classifier baseline.

Required checks before sign-off:

- Challenge description: recommended Track A baselines and metrics.
- `docs/challenge_alignment_notes.md`: literature ballparks and revised experimental sequence.
- `docs/track_a_metrics_notes.md`: required metrics.
- `docs/phase2_execution_guardrails.md`: avoid final benchmark polishing.

Rubric criteria addressed:

- Data Modeling Strategy.
- Theoretical & Analytical Justification.
- Track Selection & Problem Framing.

Stop condition:

- We have a table good enough to define the comparison floor for QRC.

Scope-control warning:

- Do not run an extensive ESN/LSTM/GARCH campaign. Phase 2 needs strong enough baselines to contextualize the QRC design. Broader ESN sweeps, GARCH-family models, and LSTM/GRU baselines are Phase 3 robustness extensions.

---

### May 24 — Milestone 4: QRC architecture specification and diagram

Goal:

- Specify the QRC architecture in enough detail that coding is implementation, not design guessing.

Required architecture fields:

- Hamiltonian/unitary: TFIM/spin-system QRC first.
- Input encoding: feature set, scaling, angle/phase/amplitude choice, direct/re-uploading density.
- Memory: rolling window, virtual nodes/re-uploading, reset policy, feedback/no-feedback.
- Readout: observables, exact expectations vs shot estimates, ridge/classical head.
- Hybrid integration: QRC state features to classical volatility/regime output.
- Resource estimate: qubits, evolution steps/depth, shots, runtime expectation.

Outputs:

- `docs/qrc_literature_design_map.md`.
- `docs/qrc_architecture_design.md`.
- `docs/qrc_architecture_diagram.mmd`.
- Initial resource table.
- Config schema for QRC experiments.

Concrete architecture decisions:

- Primary reservoir family: TFIM / spin-system QRC.
- Primary input: train-only PCA-compressed SPY+VIX volatility-state features.
- Primary input dimensions: PCA-8 first; PCA-6 fallback; PCA-10 sensitivity extension.
- Primary memory target: 40-day rolling window, motivated by the May 23 ESN result.
- First temporal implementation: compressed anchor/virtual-node injection from the 40-day window.
- First encoding: bounded angle encoding through `Ry` or `Rz`/`Ry` rotations.
- First reset policy: reset per 40-day sample/window.
- First feedback policy: no feedback.
- First readout: exact expectation values, then finite-shot estimates later.
- Observable ladder: `Z` first; `Z + X`; then `Z + X + nearest-neighbor ZZ`.
- Classical head: ridge regression on `log(future_rv_20d)`, with exponentiation before RMSE/QLIKE/MZ evaluation.
- Classical comparator: PCA-compressed ESN log-target reservoir baseline, plus persistence and HAR/Ridge/ElasticNet.

Completion note:

- Milestone 4 is complete. The QRC architecture is specified as a literature-grounded experimental design, not a final architecture claim. The design starts from a primary TFIM-QRC prototype and defines controlled probes over PCA dimension, qubit count, anchor count, observable set, evolution depth, exact versus finite-shot estimates, and later RF-QRC/onion-QRC extensions.

Required checks before sign-off:

- Challenge description: QRC Architecture Design and Quantum Platform and Resource Planning.
- `docs/theoretical_justification_notes.md`: architecture checklist and QRC property-to-signal mapping.
- `docs/platform_stakeholder_phase3_notes.md`: platform/resource planning.
- `docs/challenge_alignment_notes.md`: hardware/reservoir primers.

Rubric criteria addressed:

- QRC Architecture Design.
- Platform Justification & Resources.
- Theoretical & Analytical Justification.
- Clarity of Communication.

Stop condition:

- A judge could read the architecture note and know what we intend to execute in Phase 3.

Scope-control warning:

- Do not implement multiple QRC families yet. TFIM first; RF-QRC, onion/parallel QRC, Rydberg/analog reservoirs, amplitude encoding, and carryover-state recurrence remain fallback/extensions.

---

### May 25 — Milestone 5: First TFIM-QRC light-touch prototype

Goal:

- Execute the smallest end-to-end QRC prototype that substantiates the design claim.

Prototype target:

- Fallback prototype: 6 qubits, PCA-6 input, 6 temporal anchors, Z-only exact expectations.
- Primary prototype if feasible: 8 qubits, PCA-8 input, 8 temporal anchors, Z/X or Z/X/ZZ expectations.
- Hamiltonian: sparse or nearest-neighbor TFIM first.
- Input: train-only PCA-compressed volatility/VIX features.
- Memory: 40-day window compressed into anchor/virtual nodes.
- Output: `future_rv_20d` forecast first; `future_rv_5d` second only if the 20-day path is clean.
- Readout: ridge regression on log-volatility.
- Evaluation: RMSE, QLIKE, Mincer-Zarnowitz.

Outputs:

- Runnable QRC prototype notebook/script.
- Metrics table against at least one classical baseline, preferably the PCA-compressed ESN and HAR/Ridge.
- Saved QRC reservoir feature matrix diagnostics.
- One paragraph: what the prototype proves and does not prove.

Required checks before sign-off:

- Challenge description: light-touch prototyping guidance and QRC Architecture Design.
- `docs/qrc_architecture_design.md`: primary design and config schema.
- `docs/qrc_literature_design_map.md`: literature-to-design rationale.
- `docs/phase2_execution_guardrails.md`: Phase 2 prototype boundary.
- `docs/theoretical_justification_notes.md`: lightweight prototype probes.
- `docs/track_a_metrics_notes.md`: metrics are still central.

Rubric criteria addressed:

- QRC Architecture Design.
- Theoretical & Analytical Justification.
- Data Modeling Strategy.
- Platform Justification & Resources.

Stop condition:

- QRC runs end-to-end and produces interpretable reservoir features and volatility forecasts, even if not yet better than baselines.

Scope-control warning:

- Do not chase performance. This milestone is viability evidence and architecture validation. If the full primary prototype is slow, use the 6-qubit fallback and record the resource limitation.

---

### May 26 — Milestone 6: One focused design probe

Goal:

- Run one small experiment that directly supports a theoretical/design claim.

Choose one based on May 25 result:

- Observable probe: `Z` versus `Z + X` versus `Z + X + ZZ`.
- Memory probe: 6 versus 8 temporal anchors.
- Evolution probe: one versus two Trotter steps per anchor.
- Input probe: PCA-6 versus PCA-8.
- Resource probe: 6 versus 8 qubits.
- Shot-budget probe: exact versus 512/2048 shots.

Preferred probe if May 25 runs cleanly:

- Observable ladder probe, because it directly tests whether richer quantum/reservoir features add information beyond the simplest computational-basis readout.

Fallback probe if runtime is tight:

- PCA-6 / 6-qubit / Z-only exact expectation resource probe, with a clear statement of what prevents scaling.

Outputs:

- One compact table or plot.
- One paragraph tying result to architecture justification.
- Updated resource estimate if needed.

Required checks before sign-off:

- Challenge description: theoretical/analytical justification, resource planning, light-touch prototype boundary.
- `docs/qrc_architecture_design.md`: controlled probe plan.
- `docs/qrc_literature_design_map.md`: memory/stability/observable rationale.
- `docs/theoretical_justification_notes.md`: prototype probes.
- `docs/phase2_execution_guardrails.md`: stop rule.
- `docs/platform_stakeholder_phase3_notes.md`: resource planning and Phase 3 fallback.

Rubric criteria addressed:

- Theoretical & Analytical Justification.
- Platform Justification & Resources.
- QRC Architecture Design.

Stop condition:

- The probe either supports a design choice or gives a clear fallback for Phase 3.

Scope-control warning:

- Only one probe unless results are broken/invalid. No grids.

---

### May 27 — Milestone 7: Regime-transition interpretation layer

Goal:

- Ensure the project does not reduce to generic volatility forecasting. Build the link from volatility forecasts to regime-transition early warning.

Outputs:

- Definition of calm / turbulent / crisis or transition-risk score.
- Example timeline plot: realized volatility, forecasted volatility, and transition-warning overlay.
- At least one event-level qualitative case study.
- Clear statement that Track A metrics still evaluate the volatility-forecasting substrate.

Required checks before sign-off:

- Challenge description: Track A problem statement and stakeholder relevance.
- `docs/theoretical_justification_notes.md`: headline sub-problem and regime-transition framing.
- `docs/track_a_metrics_notes.md`: avoid replacing central metrics.
- `docs/challenge_alignment_notes.md`: regime interpretation.

Rubric criteria addressed:

- Track Selection & Problem Framing.
- Stakeholder relevance / impact.
- Clarity of Communication.
- Theoretical & Analytical Justification.

Stop condition:

- We can show how the volatility forecast becomes a regime-transition warning without pretending the warning metric replaces RMSE/QLIKE/MZ.

Scope-control warning:

- Do not invent a complex regime-labeling system. Use a reproducible, transparent definition.

---

### May 28 — Milestone 8: Platform/resource and Phase 3 execution plan

Goal:

- Make the Phase 3 plan concrete enough to satisfy the rubric.

Outputs:

- Platform table:
  - simulator type;
  - hardware/backends intended;
  - qubit range;
  - depth/evolution steps;
  - shot budgets;
  - noise/error-mitigation plan;
  - fallback path.
- Phase 3 milestone sequence.
- Risk/fallback table.
- qBraid reproducibility plan draft.

Required checks before sign-off:

- Challenge description: Quantum Platform and Resource Planning; Phase 3 Outlook; reproducibility requirements.
- `docs/platform_stakeholder_phase3_notes.md`.
- `docs/phase2_execution_guardrails.md`.
- `docs/challenge_alignment_notes.md`: qBraid readiness without credentials.

Rubric criteria addressed:

- Platform Justification & Resources.
- Phase 3 Execution Plan.
- Clarity of Communication.

Stop condition:

- A judge can see exactly what would be run in Phase 3, in what order, and what happens if hardware/simulator scaling fails.

Scope-control warning:

- Do not implement Phase 3 studies now. Only plan them and support with one small probe if already done.

---

### May 29 — Milestone 9: Submission skeleton and evidence map

Goal:

- Assemble the paper skeleton and map every claim to evidence.

Outputs:

- Paper outline with section bullets.
- Evidence table:
  - claim;
  - supporting figure/table/code;
  - source/literature reference;
  - rubric criterion;
  - status: done / weak / missing.
- List of missing items that must be closed on May 30.

Required checks before sign-off:

- Challenge description: full rubric and submission requirements.
- All created notes:
  - `docs/challenge_alignment_notes.md`
  - `docs/phase2_execution_guardrails.md`
  - `docs/theoretical_justification_notes.md`
  - `docs/track_a_metrics_notes.md`
  - `docs/platform_stakeholder_phase3_notes.md`
- Any generated result tables/plots.

Rubric criteria addressed:

- All seven rubric criteria.

Stop condition:

- The final paper can be written from the outline plus evidence table, with no new technical decisions.

Scope-control warning:

- Missing evidence can be fixed only if it is paper-critical. Otherwise mark as Phase 3.

---

### May 30 — Milestone 10: Final technical freeze

Goal:

- Freeze results, figures, claims, and Phase 3 plan. No new modeling directions.

Outputs:

- Final result tables and plots saved.
- Final architecture diagram saved.
- Final resource table saved.
- Final Phase 3 plan saved.
- Final AI-use disclosure draft.
- Final references/bibliography list.
- Paper-ready bullet draft.

Required checks before sign-off:

- Challenge description: all sections and rubric.
- `docs/phase2_execution_guardrails.md`: confirm we did not overbuild Phase 3.
- `docs/track_a_metrics_notes.md`: confirm RMSE/QLIKE/MZ are included.
- `docs/theoretical_justification_notes.md`: confirm QRC justification is not generic.
- `docs/platform_stakeholder_phase3_notes.md`: confirm stakeholder/platform/fallback sections are covered.
- `docs/challenge_alignment_notes.md`: confirm full strategy consistency.

Rubric criteria addressed:

- All seven rubric criteria.

Stop condition:

- No new experiments after this point unless they fix a factual error or broken result.

Scope-control warning:

- If something is weak, frame it as Phase 3 work. Do not start a new technical branch.

---

## May 31 — Reserved: finish paper day

Goal:

- Write and polish the final submission paper from already-frozen material.

Allowed work:

- writing;
- figure placement;
- wording cleanup;
- citation cleanup;
- final consistency pass;
- upload/package checks.

Not allowed unless absolutely necessary:

- new experiments;
- new baselines;
- new target definitions;
- new architecture variants.

## Final paper section-to-milestone mapping

1. Focus area and rationale
   - Milestones 0, 7, 9, 10.

2. Technical approach to quantum integration
   - Milestones 4, 5, 6.

3. Stakeholder relevance
   - Milestones 7, 8.

4. Data modeling strategy
   - Milestones 1, 2, 3.

5. Quantum platform justification and resource needs
   - Milestones 4, 6, 8.

6. Theoretical and analytical justification
   - Milestones 4, 5, 6, 7.

7. Phase 3 execution plan
   - Milestones 8, 9, 10.

8. Reproducibility / disclosure
   - Milestones 2, 5, 8, 10.

## Rubric sign-off checklist

Before final paper writing, each item must be at least adequate:

- QRC Architecture Design.
- Theoretical & Analytical Justification.
- Data Modeling Strategy.
- Track Selection & Problem Framing.
- Platform Justification & Resources.
- Phase 3 Execution Plan.
- Clarity of Communication.

If one is weak on May 30, fix the explanation first, not the model.
