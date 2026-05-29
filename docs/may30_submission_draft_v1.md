# May 30 — Submission Draft v1

**Working title:** Quantum Reservoir Computing for Volatility-Regime Transition Early Warning  
**Track:** A — Financial volatility prediction / regime shifts  
**Status:** first complete paper draft; requires final formatting, references, and figure/table insertion

> Note: This is a paper-ready narrative draft, not yet page-compressed. Bracketed citations should be replaced with final bibliography entries.

## Abstract / Executive Summary

We propose a hybrid quantum reservoir computing (QRC) framework for volatility-regime transition early warning from public equity-market data. Rather than treating regime detection as a standalone binary classification task, we forecast 20-day future realized volatility and derive watch, warning, and crisis-like states from the forecast path and market stress confirmations. This aligns the project with Track A evaluation requirements while preserving stakeholder interpretability for risk managers, trading desks, market makers, and volatility-sensitive portfolio managers.

Phase 2 demonstrates a working QRC volatility-forecasting pipeline, an interpretable regime-warning layer, and a hardware-aware Phase 3 plan. The final QRC prototype improves materially over an initial anchor-snapshot QRC, reducing test RMSE from 0.108 to 0.095 and increasing test correlation from 0.114 to 0.435. A strong ESN reference remains the best current forecaster, with test RMSE 0.072 and Mincer-Zarnowitz R² 0.531, so we do not claim quantum advantage in Phase 2. Instead, the classical reservoir comparison identifies a specific quantum-relevant bottleneck: the digital QRC still under-calibrates high-volatility tails. The Phase 3 plan therefore tests whether hardware-native neutral-atom/Rydberg dynamics and high-connectivity trapped-ion digital reservoirs can provide richer temporal feature maps than the shallow 6-qubit digital TFIM prototype.

## 1. Problem Framing and Track A Alignment

Financial volatility-regime transitions are difficult because they combine long memory, volatility clustering, nonlinear coupling among returns, realized volatility, VIX, drawdowns, ranges, and volume, heavy-tailed shocks, and abrupt structural changes. These features make the problem a natural candidate for reservoir computing: a nonlinear fading-memory system can transform a compact multivariate market-history sequence into a feature space where transition-relevant structure is more accessible to a simple readout.

The challenge asks for financial volatility prediction and regime-shift forecasting. We therefore use realized-volatility forecasting as the quantitative substrate and derive regime-transition warnings as an interpretation layer. The primary target is `future_rv_20d`, a 20-day future realized-volatility measure. The primary metrics are the Track A metrics: RMSE, QLIKE, and Mincer-Zarnowitz regression diagnostics. Regime-warning metrics such as q80/q90/q95 event recall and F1 are reported as secondary diagnostics, because they translate volatility forecasts into actionable risk states but do not replace the required volatility-forecasting metrics.

This framing avoids a common failure mode: treating stress-regime forecasting as a pure high/low binary classifier. A classifier may produce useful alerts, but it discards much of the information needed for volatility-sensitive risk management. By first forecasting the volatility path and then deriving watch/warning/crisis-like states, the model supports both quantitative evaluation and stakeholder interpretation.

## 2. QRC Architecture and Design Rationale

The Phase 2 architecture is a compact spin-system QRC pipeline. Public market data are converted into engineered return, realized-volatility, range, drawdown, and VIX features. Features are split chronologically into train, validation, and test periods, with scaling and PCA learned only on the training set. The final QRC uses PCA-6 market inputs, 40-day rolling windows, 10 recent temporal anchors, leaky input integration, a 6-qubit transverse-field Ising model (TFIM) reservoir, full topology, 3 Trotter steps per anchor, virtual-node trajectory features, Z/X/ZZ-style observables, train-only winsorization, top-k feature selection, and ridge readout trained on log future realized volatility.

The design follows the QRC principle that a fixed nonlinear dynamical system transforms input histories into a richer feature representation, while a simple classical readout tests whether the reservoir made the relevant signal linearly accessible. A TFIM reservoir is appropriate because volatility-regime transitions are memory-dependent and nonlinear, while spin-system reservoirs provide tunable nonlinear fading-memory dynamics through coupling strength, field strength, evolution time, input injection, virtual nodes, and measurement choices [Kutvonen et al.; Čindrak et al.; Li et al.].

The final design is also diagnostic. The initial QRC v0 used a minimal anchor-snapshot readout and produced compressed forecasts with very low test correlation. The final QRC added denser reservoir dynamics, virtual-node trajectory features, leaky input integration, and calibrated readout selection. These changes were not arbitrary hyperparameter tuning; they were motivated by the observed gap between the digital QRC and a classical ESN reservoir. The ESN showed that the task benefits from nonlinear trajectory processing. The leaky-input QRC variant partially restored missing trajectory-shape information and improved high-volatility responsiveness.

**Figure placeholder:** `docs/qrc_architecture_paper_figure.mmd` or rendered PNG/SVG.  
**Caption:** Hybrid QRC volatility-warning pipeline. Public market features are converted into rolling temporal inputs, encoded into a compact TFIM quantum reservoir, read out by a ridge model for realized-volatility forecasting, and translated into forecast-aware regime-warning states.

## 3. Data, Metrics, and Baselines

The data pipeline uses public equity-market data, including SPY/OHLCV-style information and VIX-derived market-stress features. The target is 20-day future realized volatility. The split is chronological, not randomized, to avoid look-ahead leakage and to mimic realistic forecasting conditions.

The model is evaluated with the three required Track A metrics:

- **RMSE**, measuring standard forecast error;
- **QLIKE**, a volatility-specific quasi-likelihood loss used for volatility forecast comparison [Patton];
- **Mincer-Zarnowitz regression diagnostics**, measuring forecast calibration and efficiency.

The paper also reports derived regime-warning diagnostics. These are not substitutes for RMSE/QLIKE/MZ. They answer the stakeholder question: does the forecast path produce useful early warning states around high-volatility periods?

Classical baselines include persistence/naive forecasts, HAR-like ridge regression, and an ESN reference reservoir. The ESN is intentionally strong. It is not used as a weak strawman; it tests whether reservoir computing is a good modeling class for volatility dynamics and gives a performance target for the QRC variants.

## 4. Phase 2 Results

### 4.1 Forecasting performance

**Table 1. Canonical model comparison.**

| Model | Test RMSE | Test QLIKE | Test MZ R² | Test corr. | Test pred. std. | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| QRC v0 anchor snapshot | 0.107953 | -1.863498 | 0.013055 | 0.114259 | 0.021781 | minimal initial QRC; strongly compressed forecasts |
| Final QRC encoding/readout | 0.095082 | -2.229724 | 0.188898 | 0.434624 | 0.054796 | improved QRC; nontrivial signal but still under-calibrated |
| ESN reference reservoir, n=500 | 0.071896 | -2.504775 | 0.531270 | 0.728883 | 0.080431 | strongest classical reservoir reference |

The final QRC improves substantially over the initial QRC. Test RMSE improves from 0.108 to 0.095, test correlation rises from 0.114 to 0.435, and forecast variance increases from 0.022 to 0.055. This shows that the quantum-reservoir pipeline is not merely producing a near-constant forecast; it extracts nontrivial volatility signal after the trajectory-aware encoding and readout improvements.

However, the ESN remains the strongest current forecaster. This is important and should not be hidden. The final QRC's lower Mincer-Zarnowitz R² relative to ESN indicates remaining calibration and high-volatility tail-amplitude limitations. We therefore do not claim completed quantum advantage in Phase 2. Instead, we use the ESN comparison to identify the remaining bottleneck: richer within-window trajectory processing and tail calibration.

**Figure placeholder:** `results/figures/phase2_qrc_progress_regime_ablation.png`.  
**Caption:** Reservoir progress from initial QRC to final QRC and ESN reference. The initial QRC prototype used a minimal anchor-snapshot readout and produced compressed forecasts. Iterative QRC development improved forecast error, warning-layer F1, and crisis-like responsiveness. The ESN reference validates the reservoir-computing architecture and defines the remaining Phase 3 calibration target.

### 4.2 Regime-warning interpretation

The forecast-aware regime layer maps predicted volatility and transparent stress confirmations into normal, watch, warning, and crisis-like states. This layer is designed for interpretability: it does not hide the regression model behind a black-box classifier, and it preserves direct links to volatility level, forecast slope/persistence, and market-stress confirmations.

The q80 watch+ layer is the most mature Phase 2 warning level. The final QRC reaches useful q80 watch+ performance and approaches the ESN on this specific interpretive layer, while q95 crisis-like detection remains difficult. This is expected: the most extreme volatility tails are rare, noisy, and exactly where the QRC still shows amplitude compression. The COVID-window case study is therefore used qualitatively: it shows whether forecasts and warning states respond in the correct period, but it is not presented as proof of deployable crisis prediction.

**Table placeholder:** compact regime-warning table with q80/q90/q95 F1 and recall for QRC v0, final QRC, and ESN.  
**COVID case-study placeholder:** one row or one mini-panel if space permits.

## 5. Value of the Quantum Approach Compared with Classical Alternatives

The value of the quantum approach is evaluated against strong classical alternatives, not assumed. In Phase 2, the ESN remains the strongest forecaster, so the current result is not a performance-based quantum advantage. The quantum-relevant value is mechanistic and diagnostic.

First, the problem structure is well matched to reservoir computing. Volatility-regime transitions require nonlinear fading memory over multiscale market histories. QRC provides a physical-reservoir route to such temporal feature maps, using tunable Hamiltonian dynamics and measured observables rather than a fully trained recurrent network.

Second, the experiments identify why the current digital QRC underperforms. The initial QRC was too compressed. The final QRC improved after adding trajectory-aware leaky input integration and richer observable/readout structure. This shows that the quantum reservoir is sensitive to the same within-window temporal structure that makes the ESN successful, but the current shallow 6-qubit digital reservoir remains less expressive and less well calibrated in high-volatility tails.

Third, the remaining gap points to a specific Phase 3 quantum hypothesis. If the bottleneck is feature expressivity and tail-amplitude calibration, then the strongest next test is not a larger classical readout. It is a richer physical reservoir. Hardware-native neutral-atom/Rydberg dynamics can generate correlated many-body features without digitally compiling every interaction into deep gate sequences. High-connectivity trapped-ion hardware can test whether dense digital QRC circuits benefit from reduced routing overhead. IBM/Qiskit provides a reproducible superconducting benchmark for resource and accessibility comparison.

Thus, Phase 2 demonstrates quantum value as a disciplined design-and-diagnosis program: the QRC architecture is executable, improves under theoretically motivated modifications, and yields a concrete, hardware-specific Phase 3 test of whether native quantum reservoir dynamics can close the gap to strong classical reservoirs.

## 6. Platform, Resource, and Phase 3 Plan

**Table 2. Compact hardware-resource plan.**

| Platform | Role | Why needed | Resource scale | Fallback |
|---|---|---|---|---|
| QuEra/Aquila neutral atoms | Primary hardware-native QRC route | Rydberg/Ising many-body dynamics can act directly as analog reservoir | 16-50 atom pilot; 50-100+ atoms if feasible; 500-2000 shots/program | emulator-only; simplified geometry; digital QRC comparator |
| IonQ trapped ions | Dense digital-QRC comparator | high connectivity tests whether compact dense QRC avoids superconducting routing burden | 6-10 qubits; 1-4 anchors; 1-2 steps; 500-4000 shots/circuit | shot simulator; IBM resource benchmark |
| IBM/Qiskit superconducting | Accessibility/reproducibility benchmark | direct Qiskit path; fake-backend resource estimates; limited QPU validation | 6-10 qubits; 1024-4096 shots for selected circuits | fake backend only; reduce interaction density |

The resource plan is deliberately staged. The full final simulator configuration is not submitted naively to QPU hardware. A 6-qubit full-topology ZZ layer has 15 pair interactions, approximately 30 two-qubit gates per Trotter step before routing. With 10 anchors and 3 steps per anchor, the naive full digital version would require roughly 900 two-qubit gates before routing. Phase 3 therefore uses reduced representative circuits for QPU validation and reserves full sweeps for simulators.

Phase 3 has six concrete steps: freeze the data and baselines; characterize representative digital QRC circuits; run targeted trapped-ion digital validation; translate the TFIM-QRC design into neutral-atom/Rydberg analog reservoirs; validate the neutral-atom reservoir on emulator and QPU if access permits; compare analog neutral-atom, trapped-ion digital, and IBM/superconducting benchmark paths. All backend runs will save qubit/atom counts, depth or analog evolution schedule, native gates or pulse metadata, shots, seeds, package versions, raw counts/bitstrings, derived features, and downstream metrics.

This plan directly addresses the hardware-resource rubric: QuEra/Aquila is the primary scientific platform, IonQ is the connectivity comparator, and IBM/Qiskit is the accessible reproducibility benchmark.

## 7. AI-Use and Reproducibility Disclosure Draft

We used AI assistance as a development and documentation aid for literature organization, experiment planning, code scaffolding, result interpretation, and draft preparation. All modeling decisions, code execution, metric generation, result selection, and final claims were reviewed by the project author. The final submission is based on reproducible scripts, notebooks, saved metrics tables, and figures in the project repository. AI assistance was not used to fabricate results or replace empirical evaluation.

The repository will include deterministic preprocessing, chronological splits, saved model configurations, export scripts, result tables, and backend metadata. QPU-facing Phase 3 cells will be isolated so that the workflow can run in simulator-only mode without private credentials.

## 8. Limitations and Phase 3 Work

Phase 2 does not show quantum advantage. The ESN remains the strongest current forecaster. The final QRC also remains under-calibrated in high-volatility tails, as reflected by lower Mincer-Zarnowitz R² and weaker crisis-like response. These limitations are not ignored; they define the Phase 3 plan.

Phase 3 will test whether richer reservoir dynamics can reduce this gap. The primary route is neutral-atom/Rydberg analog QRC, because native many-body dynamics are the most direct physical match to the reservoir-computing hypothesis. The secondary route is trapped-ion digital QRC, because high connectivity may better support dense compact reservoir circuits. The reproducibility benchmark is IBM/Qiskit, because it provides the broadest accessible path for resource characterization and comparison.

No further Phase 2 modeling branches should be opened unless they correct a factual error or broken result. Additional improvements such as full LSTM/GARCH comparison, full noisy simulation, larger QRC sweeps, and live QPU execution are Phase 3 work.

## 9. Conclusion

This Phase 2 project produced a working QRC volatility-forecasting pipeline, evaluated it with the required Track A metrics, compared it against strong classical baselines, and connected the forecast output to an interpretable regime-warning layer. The final QRC does not yet beat the strongest ESN baseline, but it improves substantially over the initial QRC and reveals a specific, testable bottleneck: reservoir expressivity and tail-amplitude calibration.

That diagnosis is the core Phase 3 opportunity. The next stage should test whether hardware-native neutral-atom/Rydberg reservoirs and high-connectivity trapped-ion digital reservoirs can provide richer temporal feature maps than the shallow digital TFIM prototype, while IBM/Qiskit remains the accessible benchmark for reproducibility and resource accounting.

## Reference placeholders

- Li et al., QRC for realized-volatility forecasting, arXiv:2505.13933.
- Kutvonen et al., Scientific Reports TFIM/QRC paper.
- Čindrak et al., QRC memory/nonlinearity trade-off paper.
- Jaeger or Lukoševičius, ESN/reservoir computing foundation.
- Patton, QLIKE / volatility forecast comparison.
- Antoncich et al., neutral-atom / Aquila QRC reference.
- QuEra/Aquila documentation or AWS Braket Aquila documentation.
- IonQ/trapped-ion hardware/connectivity documentation.
- IBM/Qiskit fake backend / Runtime documentation if needed.
