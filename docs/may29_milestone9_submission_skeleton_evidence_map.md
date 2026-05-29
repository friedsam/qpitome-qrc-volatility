# May 29 — Milestone 9: Submission Skeleton and Evidence Map

Date: May 29, 2026  
Track: A — Financial volatility and regime-transition forecasting  
Status: working submission assembly map

## 1. Purpose

Milestone 9 converts the technical work into a final-paper plan. The objective is to ensure that every important claim in the submission has supporting evidence, maps to a rubric criterion, and can be written without reopening technical design decisions.

Stop condition for this milestone:

- the final 3-page paper can be drafted directly from this outline and evidence table;
- missing items are either closed on May 30 or marked explicitly as Phase 3;
- no new modeling or hardware decisions are introduced unless they are paper-critical.

## 2. Submission strategy

The final paper must be written in an advancement-oriented style:

- lead with the strongest evidence and Phase 3 upside;
- show that Phase 2 produced a working, improved QRC pipeline;
- present limitations as specific Phase 3 work packages, not as apologies;
- avoid overclaiming quantum advantage;
- make the hardware/resource plan concrete enough for the rubric.

Internal truth:

- final QRC is not yet ESN-level;
- digital QRC still compresses tail amplitudes;
- crisis-like detection depends on stress confirmations;
- neutral-atom translation remains Phase 3 work.

External submission story:

- QRC is viable and improved materially from v0 to final QRC;
- the regime-warning layer makes the volatility forecast actionable;
- ESN validates the reservoir-computing direction and defines a performance target;
- neutral atoms provide the strongest hardware-native Phase 3 route;
- IonQ and IBM provide targeted comparison and reproducibility paths.

## 3. Proposed 3-page paper outline

### Title

**Quantum Reservoir Computing for Volatility-Regime Transition Early Warning**

Subtitle option:

**A Phase 2 QRC prototype with reservoir-progress evidence and a hardware-native Phase 3 plan**

### Abstract / executive summary paragraph

- Track A problem: public equity-market data, volatility forecasting, regime-transition early warning.
- Proposed solution: hybrid QRC pipeline that forecasts 20-day realized volatility and converts forecasts into watch/warning/crisis-like risk states.
- Phase 2 evidence: QRC v0 -> final QRC improvement, comparison to persistence/HAR-like/ESN, interpretable regime-warning layer.
- Phase 3 plan: neutral-atom analog QRC as primary hardware-native route; IonQ trapped-ion digital comparator; IBM/Qiskit reproducibility benchmark.

### Section 1 — Problem and Track A alignment

Core bullets:

- Volatility-regime transitions are nonlinear, multiscale, weakly predictable, and economically important.
- The challenge asks for volatility forecasting / regime-shift forecasting, not only classification.
- We therefore forecast `future_rv_20d` and derive regime-transition warnings from the forecast and market stress confirmations.
- Stakeholders: trading desks, market makers, portfolio/risk managers, derivatives desks, clearing/risk-control teams.

Evidence to use:

- `docs/challenge_alignment_notes.md`
- `docs/theoretical_justification_notes.md`
- `docs/track_a_metrics_notes.md`
- regime-warning figures/tables from Milestone 7

### Section 2 — QRC architecture and design rationale

Core bullets:

- Reservoir: compact TFIM/spin-system QRC.
- Input: PCA-6 market-history features derived from public equity/VIX-style data.
- Memory: 40-day rolling windows, temporal anchors, leaky input integration in final model.
- Dynamics: 6-qubit TFIM reservoir, full topology in final configuration, 3 Trotter steps per anchor, virtual nodes.
- Readout: measured/derived observable features into ridge regression on log volatility target.
- Regime interpretation: forecasted volatility plus transparent stress confirmations generate watch/warning/crisis-like states.

Evidence to use:

- `scripts/export_qrc_v0_predictions.py`
- `scripts/export_final_qrc_predictions.py`
- `src/qpitome_qrc/qrc/tfim_reservoir.py`
- QRC v0 / final QRC prediction export metrics
- QRC progress ablation notebook

### Section 3 — Data and evaluation strategy

Core bullets:

- Public market-data pipeline; chronological train/validation/test split.
- Primary target: 20-day future realized volatility.
- Required metrics: RMSE, QLIKE, Mincer-Zarnowitz / MZ R².
- Secondary diagnostics: q80/q90/q95 event/warning metrics and regime-layer F1.
- Classical baselines: persistence, HAR-like ridge, ESN reference.

Evidence to use:

- `docs/track_a_metrics_notes.md`
- baseline metrics table
- QRC final export metrics
- ESN export metrics
- regime-layer comparison notebook

### Section 4 — Results: QRC progress and reservoir-computing ceiling

Core bullets:

- QRC v0 was a minimal 6-feature anchor-snapshot prototype and was strongly amplitude-compressed.
- Final QRC improved test RMSE by about 12% relative to v0 and improved correlation substantially.
- Final QRC still trails ESN, but ESN validates the reservoir-computing direction and provides a Phase 3 performance target.
- The last progress plot should be the main results figure.

Numbers to use:

| Model | Test RMSE | Test corr | Test pred_std | Interpretation |
|---|---:|---:|---:|---|
| QRC v0 | 0.107953 | 0.114259 | 0.021781 | minimal initial QRC, heavily compressed |
| Final QRC | 0.095082 | 0.434624 | 0.054796 | improved quantum reservoir prototype |
| ESN | 0.071896 | 0.728883 | 0.080431 | strong classical reservoir reference |

Main results sentence:

> Iterative QRC development reduced forecast error and increased stress-period responsiveness. The ESN reference shows remaining headroom, while the final QRC demonstrates that the quantum-reservoir pipeline produces nontrivial, improving volatility signal.

Evidence to use:

- `results/tables/phase2_qrc_progress_regime_ablation_summary.csv`
- `results/figures/phase2_qrc_progress_regime_ablation.png`
- `results/figures/phase2_qrc_progress_forecast_timeline.png`
- `scripts/export_qrc_v0_predictions.py`
- `scripts/export_final_qrc_predictions.py`
- `scripts/export_esn_predictions.py`

### Section 5 — Regime-warning layer and stakeholder value

Core bullets:

- Volatility forecasts become actionable when converted into risk states.
- A transparent forecast-aware layer maps model outputs and market confirmations into normal/watch/warning/crisis-like states.
- This layer supports stakeholder interpretation without claiming deployable trading alpha.
- q80 warning performance is useful; q95 crisis-like remains hard and becomes a Phase 3 calibration target.

Evidence to use:

- `notebooks/phase2_forecast_aware_regime_layer.ipynb`
- `notebooks/phase2_forecast_aware_regime_layer_comparison.ipynb`
- regime count/evaluation tables
- COVID-window example table

### Section 6 — Hardware/resource plan and Phase 3 execution

Core bullets:

- Primary hardware-native route: QuEra/Aquila-class neutral atoms because QRC is naturally an analog many-body reservoir problem.
- Scientific comparator: IonQ-class trapped ions to test whether high connectivity benefits compact dense digital QRC.
- Accessibility/reproducibility benchmark: IBM/Qiskit superconducting path for resource estimates and limited QPU feasibility.
- Estimates: 6-10 qubit digital QPU pilots; 16-50 atom neutral-atom pilot; 500-4000 shots depending backend; reduced representative windows before any full-scale run.
- qBraid plan: simulator-first, backend-specific cells isolated, metadata saved for all runs.

Evidence to use:

- `docs/may28_milestone8_platform_phase3_plan.md`
- `docs/platform_stakeholder_phase3_notes.md`
- `docs/phase2_execution_guardrails.md`

### Section 7 — Phase 3 deliverables and close

Core bullets:

- Freeze data and baselines.
- Characterize representative circuits.
- Run targeted digital QPU validation.
- Translate TFIM-QRC into neutral-atom analog QRC.
- Compare analog neutral-atom, trapped-ion digital, and superconducting benchmark paths.
- Final output: qBraid-executable workflow, backend metadata, reproducible figures/tables, Phase 3 paper.

Closing sentence:

> Phase 2 demonstrates a viable, improving QRC volatility-warning pipeline. Phase 3 is justified because the remaining gap is now specific and testable: whether hardware-native reservoir dynamics, especially neutral-atom/Rydberg dynamics, can reduce the tail-amplitude and expressivity limitations observed in the digital simulator prototype.

## 4. Figure and table plan

### Main Figure 1 — QRC progress and reservoir ceiling

Use:

- `results/figures/phase2_qrc_progress_regime_ablation.png`
- optionally pair with `results/figures/phase2_qrc_progress_forecast_timeline.png` if space allows.

Purpose:

- show QRC v0 -> final QRC -> ESN progress;
- show final QRC is not finished but is moving in the right direction;
- make the Phase 3 case visually.

Status: done.

### Main Table 1 — Forecast metrics and regime-warning diagnostics

Use compact table:

| Model | RMSE | QLIKE | MZ R² | q80 watch+ F1 | q90 warning+ F1 | q95 crisis-like F1 |
|---|---:|---:|---:|---:|---:|---:|
| QRC v0 | from export | from export | from export | from progress notebook | from progress notebook | from progress notebook |
| Final QRC | from export | from export | from export | from progress notebook | from progress notebook | from progress notebook |
| ESN | from export | from export | from export | from progress notebook | from progress notebook | from progress notebook |

Status: done but needs final paper formatting.

### Main Table 2 — Hardware resource plan

Use compact version of `docs/may28_milestone8_platform_phase3_plan.md`:

| Platform | Role | Why needed | Resource scale | Fallback |
|---|---|---|---|---|
| QuEra/Aquila | primary analog QRC | native many-body reservoir | 16-50 atom pilot; 500-2000 shots/program | emulator / simplified geometry |
| IonQ | dense digital comparator | high connectivity for compact QRC | 6-10 qubits; 500-4000 shots/circuit | shot simulator / IBM benchmark |
| IBM/Qiskit | accessibility benchmark | Qiskit reproducibility and resource estimates | 6-10 qubits; 1024-4096 shots for limited QPU check | fake backend only |

Status: done but may need page compression.

### Optional Figure — Architecture diagram

Needed if the final paper feels too text-heavy.

Diagram:

```text
Public market data -> feature/PCA window -> QRC reservoir -> volatility forecast -> regime-warning layer -> stakeholder signal
```

Status: weak/missing. Can be generated quickly on May 30 if needed.

## 5. Evidence map

| Claim | Supporting evidence | Literature/source reference | Rubric criterion | Status |
|---|---|---|---|---|
| Track A should be framed as realized-volatility forecasting with derived regime-transition warning, not pure classification. | `docs/challenge_alignment_notes.md`; `docs/track_a_metrics_notes.md`; use of `future_rv_20d` target | Challenge Track A description; volatility forecasting literature | Track Selection & Problem Framing; Data Modeling Strategy | done |
| RMSE, QLIKE, and Mincer-Zarnowitz are central evaluation metrics. | `docs/track_a_metrics_notes.md`; model export metrics tables | Challenge Track A metrics; Patton QLIKE reference if cited | Data Modeling Strategy | done |
| Volatility-regime transitions are suitable for reservoir computing because they involve nonlinear, multiscale, memory-dependent dynamics. | `docs/theoretical_justification_notes.md` | QRC/realized-volatility literature; volatility clustering/regime-switching references | Theoretical & Analytical Justification | done |
| The Phase 2 QRC architecture is explicit and executable. | `src/qpitome_qrc/qrc/tfim_reservoir.py`; export scripts; notebooks | QRC spin-system / TFIM references | QRC Architecture Design | done |
| QRC v0 was a minimal prototype and final QRC materially improved. | `scripts/export_qrc_v0_predictions.py`; `scripts/export_final_qrc_predictions.py`; progress plot/table | Internal results | QRC Architecture Design; Clarity of Communication | done |
| Final QRC captures nontrivial volatility signal but still trails ESN. | final QRC metrics; ESN metrics; progress plot | ESN / reservoir-computing baseline literature | Data Modeling Strategy; Clarity of Communication | done |
| ESN validates the reservoir-computing direction and defines a Phase 3 performance target. | `scripts/export_esn_predictions.py`; comparison notebook | Reservoir computing literature | Theoretical & Analytical Justification; Data Modeling Strategy | done |
| Forecasts can be translated into actionable regime-warning states. | `notebooks/phase2_forecast_aware_regime_layer.ipynb`; comparison notebook; regime tables | Risk-management / volatility-regime interpretation | Track Selection & Problem Framing; Clarity | done |
| q80/q90 warnings are promising; q95 crisis-like warning remains difficult. | regime-layer metrics; COVID-window table | Internal results | Data Modeling Strategy; Phase 3 Execution Plan | done |
| QuEra/Aquila is the primary Phase 3 hardware target because analog Rydberg dynamics match the QRC reservoir hypothesis. | `docs/may28_milestone8_platform_phase3_plan.md` | QuEra/Aquila / neutral-atom QRC references; challenge tutorials | Platform Justification & Resources; Phase 3 Plan | done but cite needs final reference |
| IonQ is a targeted digital comparator because high connectivity may help dense QRC circuits. | `docs/may28_milestone8_platform_phase3_plan.md` | trapped-ion connectivity / IonQ reference | Platform Justification & Resources | done but cite needs final reference |
| IBM/Qiskit is an accessibility benchmark and reproducibility path. | `docs/may28_milestone8_platform_phase3_plan.md`; free/limited IBM path | Qiskit/IBM docs; fake backend notes | Platform Justification & Resources; Reproducibility | done |
| Full noisy simulation and full QPU scaling belong to Phase 3, not Phase 2. | `docs/phase2_execution_guardrails.md`; platform notes | Challenge Phase 2 vs Phase 3 requirements | Phase 3 Execution Plan | done |
| qBraid reproducibility is planned with backend metadata, isolated QPU cells, and simulator-first execution. | `docs/may28_milestone8_platform_phase3_plan.md`; `docs/platform_stakeholder_phase3_notes.md` | qBraid challenge requirements | Platform Resources; Phase 3 Plan | done |
| Stakeholder value is risk-warning support, not deployable trading alpha. | `docs/platform_stakeholder_phase3_notes.md`; regime layer | Challenge stakeholder context | Track Framing; Clarity | done |
| The project is ready for Phase 3 because the remaining gap is specific and testable. | QRC progress plot; hardware plan; missing-items table | Internal results and platform plan | Phase 3 Execution Plan; Clarity | done |

## 6. Rubric coverage map

| Rubric criterion | Best evidence | Status | May 30 action |
|---|---|---|---|
| QRC Architecture Design | TFIM-QRC implementation; final QRC config; architecture bullets | strong | Compress into one clear paragraph and/or diagram |
| Theoretical & Analytical Justification | volatility-memory/nonlinearity argument; QRC fading-memory feature map; hardware-native QRC thesis | strong | Add 2-3 final citations |
| Data Modeling Strategy | `future_rv_20d`, chronological splits, RMSE/QLIKE/MZ, baselines | strong | Finalize compact result table |
| Track Selection & Problem Framing | regime-transition early warning from volatility forecast | strong | Make first paragraph very direct |
| Platform Justification & Resources | May 28 hardware plan with QuEra/IonQ/IBM roles and estimates | adequate/strong | Tighten to fit 3 pages; add exact citations |
| Phase 3 Execution Plan | staged Phase 3 milestones and fallback table | strong | Convert to concise table |
| Clarity of Communication | progress plot; timeline plot; submission-ready paragraphs | strong | Select final figures and captions |

## 7. Missing / weak items to close on May 30

### Paper-critical

1. **Final 3-page narrative compression.**  
   Status: missing.  
   Action: write full submission draft using this skeleton.  
   Deadline: May 30.

2. **Final figure/table selection.**  
   Status: partly done.  
   Action: choose either one progress plot only, or progress plot + compact hardware table.  
   Deadline: May 30.

3. **Exact final metrics table.**  
   Status: done in outputs but needs paper-ready formatting.  
   Action: copy from `phase2_qrc_progress_regime_ablation_summary.csv` and final export metrics.  
   Deadline: May 30.

4. **Final citations.**  
   Status: weak.  
   Action: choose minimal set: QRC/realized-volatility paper, QLIKE/Patton reference, ESN/reservoir-computing reference, QuEra/Aquila or neutral-atom QRC reference, IonQ/trapped-ion connectivity reference, IBM/Qiskit/fake backend docs if needed.  
   Deadline: May 30.

5. **Architecture diagram if space permits.**  
   Status: optional/weak.  
   Action: include only if it improves clarity without crowding results.  
   Deadline: May 30.

### Not paper-critical; mark as Phase 3

1. Full hardware execution on qBraid/QPU.
2. Full noisy simulation or density-matrix scaling study.
3. Full GARCH/LSTM implementation.
4. Full neutral-atom Rydberg encoding implementation.
5. Full IonQ conversion/execution pipeline.
6. Final production-grade regime classifier.

## 8. Final-paper drafting order for May 30

1. Write 150-word executive summary.
2. Insert main progress figure and caption.
3. Insert compact result table.
4. Write architecture/data paragraph.
5. Write regime-warning/stakeholder paragraph.
6. Insert compact hardware/resource table.
7. Write Phase 3 closing paragraph.
8. Add minimal citations.
9. Cut aggressively to 3 pages.

## 9. Recommended final claims

Use these claims in the paper:

- We built a working hybrid QRC volatility-forecasting pipeline.
- We moved from a minimal QRC v0 to a materially stronger final QRC prototype.
- We evaluate the required volatility metrics and use regime-warning metrics as interpretation.
- The ESN result validates the reservoir-computing direction and defines a strong classical target.
- The regime-warning layer turns forecasts into actionable risk states.
- Neutral atoms are the primary Phase 3 hardware-native route because QRC is naturally an analog reservoir problem.
- IonQ tests whether high connectivity benefits compact dense digital QRC.
- IBM/Qiskit provides the accessible reproducibility benchmark.
- The remaining performance gap is specific, measurable, and suitable for Phase 3.

Avoid these claims:

- We have demonstrated quantum advantage.
- QRC already beats all classical baselines.
- The model is ready for trading or deployment.
- Full hardware validation has been completed.
- Crisis detection is solved.
