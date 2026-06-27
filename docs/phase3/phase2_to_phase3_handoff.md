# Phase 2 to Phase 3 Handoff

**Project:** Qpitome QRC Volatility  
**Track:** Financial volatility prediction / regime-transition early warning  
**Purpose:** Preserve the Phase 2 scientific record in a compact, usable form before creating a cleaner Phase 3 branch.

## 1. Executive Summary

Phase 2 produced a working volatility-forecasting pipeline, strong classical baselines, a digital TFIM-style QRC prototype, and an interpretable regime-warning layer. The project moved from an initial stress-classification idea to the stronger framing of **continuous 20-day realized-volatility forecasting** with derived warning states.

The main Phase 2 lesson is specific: the compact digital QRC does extract nontrivial volatility signal, but it remains **under-dispersed** in high-volatility tails. The ESN baseline showed that reservoir computing is a good model class for this task; the QRC development showed that trajectory-aware encoding and richer readout improve the quantum reservoir substantially. Phase 3 should therefore test whether **hardware-native many-body dynamics**, especially neutral-atom/Rydberg dynamics, can provide richer reservoir features than the shallow digital TFIM prototype.

## 2. Phase 2 Problem Framing

The task is not generic market prediction. It is volatility-regime transition early warning.

Phase 2 settled on this formulation:

- Forecast target: `future_rv_20d`, annualized 20-day forward realized volatility.
- Main evaluation: RMSE, QLIKE, and Mincer-Zarnowitz regression diagnostics.
- Secondary interpretation: translate forecasts into `normal`, `watch`, `warning`, and `crisis_like` states using train-calibrated thresholds and stress confirmations.
- Stakeholder use: risk escalation, hedging review, trading-desk monitoring, market-making controls.

This avoids the weaker version of the task: a hand-labeled binary stress classifier. The regression target preserves more information and still allows regime warnings to be derived.

## 3. Data Pipeline

Phase 2 used public SPY OHLCV-style data plus VIX history. The processed modeling frame spans roughly 1993-2024 and is split chronologically:

| Split | Period | Purpose |
|---|---|---|
| Train | 1993-2014 | fit scalers, PCA, thresholds, readouts |
| Validation | 2015-2019 | tune/select models |
| Test | 2020-2024 | final evaluation, including COVID, 2022 rate-hike stress, 2023 banking stress |

The feature set captures:

- returns and absolute/squared returns;
- high-low and open-close range;
- Parkinson and Garman-Klass variance proxies;
- trailing realized volatility over multiple horizons;
- volatility ratios/slopes;
- drawdown;
- VIX level, range, changes, and rolling statistics;
- volume/liquidity stress.

Design constraint: **all preprocessing must be train-only**. Scaling, PCA, thresholds, feature selection, and readout fitting are learned on the training split and then applied unchanged.

## 4. Baselines

The baselines became part of the grounding system for the project.

| Baseline | Role | Main takeaway |
|---|---|---|
| Persistence | volatility-clustering floor | useful but weak |
| HAR / ridge / ElasticNet | econometric/statistical floor | strong low-cost comparator |
| ESN reservoir | classical reservoir ceiling | confirms within-window trajectory processing matters |

The ESN result was especially important. It showed that a reset-window reservoir can outperform static-feature baselines, so the task benefits from nonlinear temporal processing. This made the QRC failure mode diagnosable rather than vague.

Representative final Phase 2 comparison:

| Model | Test RMSE | Test QLIKE | MZ R² | Corr. | Pred. std. | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| QRC v0 | ~0.108 | ~-1.863 | ~0.013 | ~0.114 | ~0.022 | minimal QRC, highly compressed |
| Final QRC | ~0.093-0.095 | ~-2.23 to -2.27 | ~0.19-0.22 | ~0.43-0.47 | ~0.055-0.058 | meaningful improvement, still compressed |
| ESN | ~0.072 | ~-2.505 | ~0.531 | ~0.729 | ~0.080 | strongest current reservoir reference |
| HAR-ridge | ~0.101 | ~-2.079 | ~0.354 | — | — | strong classical/econometric floor |

The exact QRC number differs slightly between the milestone docs and final paper draft because later writeup versions used the stronger top-540/all-pairs readout result. The important direction is stable: QRC improved materially, but ESN remains stronger.

## 5. Digital QRC Development

The QRC architecture evolved through controlled probes rather than random tuning.

Final Phase 2 QRC family:

- 6-qubit TFIM-style reservoir;
- PCA-6 market inputs;
- 40-day rolling windows;
- recent-biased temporal anchors;
- leaky trajectory integration before encoding;
- full/dense ZZ interaction topology;
- transverse X rotations;
- deterministic disorder fixed across samples;
- virtual-node readout during the trajectory;
- Z, X, and ZZ observable features;
- train-only feature selection / winsorization;
- ridge readout on log volatility.

Key design lessons:

| Probe / idea | Outcome | Lesson |
|---|---|---|
| Sparse anchor snapshots | weak, under-dispersed | QRC was not seeing enough trajectory shape |
| Z vs ZX vs ZXZZ observables | richer readout helped | correlations matter for stress states |
| Full topology / disorder | helped feature diversity | dense interactions better match multi-indicator stress |
| Virtual nodes | helped feature richness | time-resolved reservoir states matter |
| Leaky input integration | major improvement | trajectory representation was the main bottleneck |
| Polynomial readout | not decisive | bottleneck is reservoir features, not just readout nonlinearity |
| Repeated re-encoding | degraded in tested setting | more encoding is not automatically better |
| Synthetic feature noise | small metric effect | final QRC features were not extremely fragile |

## 6. Main Failure Mode

The final QRC is not a toy, but it is not yet ESN-level.

The core failure mode is:

> QRC forecasts remain too compressed in high-volatility tails.

Evidence:

- QRC prediction standard deviation remains below ESN prediction standard deviation.
- QRC improves q80 warning behavior but struggles more at q90/q95.
- During COVID-like crisis windows, realized volatility reaches extreme levels while QRC predictions rise but remain below the highest thresholds.

This is a useful failure mode because it points directly to Phase 3: richer reservoir dynamics, better tail calibration, and hardware-native many-body feature generation.

## 7. Regime-Warning Layer

Phase 2 added an interpretation layer so the output is not just a scalar volatility forecast.

Definitions:

- q80 event: high-volatility event;
- q90 event: extreme-volatility event;
- q95 event: crisis-candidate event.

Warning states combine forecast information with transparent market-stress confirmations such as VIX level/range, drawdown, selloff, volume stress, and realized-volatility acceleration.

The layer is useful because it makes the project stakeholder-facing without inventing opaque regime labels. It also makes the QRC failure mode operationally visible: broad warnings can work even when crisis-tail amplitude calibration remains weak.

## 8. Phase 3 Implications

Phase 3 should keep two tracks separate.

### Foundation Track

Purpose: preserve a reproducible, judge-safe baseline.

Must keep:

- frozen data pipeline;
- chronological splits;
- train-only preprocessing;
- persistence / HAR / ESN baselines;
- final digital QRC reference;
- regime-warning evaluation;
- qBraid-compatible run instructions.

### Bold Track

Purpose: test the quantum-relevant hypothesis.

Hypothesis:

> Native neutral-atom/Rydberg many-body dynamics may generate richer nonlinear reservoir features than the compact digital TFIM-QRC, potentially reducing the tail-compression bottleneck.

First Rydberg study should be modest:

- select representative calm/watch/warning/crisis-like windows;
- use emulator/simulator first;
- keep atom geometry and encoding simple;
- extract bitstring statistics, local excitations, pair correlations, and distribution summaries;
- test feature stability and regime separability;
- use QPU only as a small validation subset if access appears.

Do not claim quantum advantage unless the evidence supports it. The defensible initial claim is hardware-native reservoir feasibility and feature sensitivity, not ESN-beating performance.

## 9. Suggested Phase 3 Workstreams

| Workstream | Goal | Minimum output |
|---|---|---|
| Reproducibility freeze | ensure Phase 2 results still run | tests + reference metric tables |
| Representative windows | define the experimental subset | `phase3_representative_windows.csv` |
| Rydberg formulation | translate QRC hypothesis to neutral atoms | geometry, encoding, pulse/evolution, features |
| Emulator probe | test whether features separate regimes | feature table + separability/stability metrics |
| Optional QPU pilot | compare hardware to emulator | small bitstring/counts feature-stability report |
| Final cleanup | submission-ready branch | clean README, runbook, docs, result tables |

## 10. Evidence Map

| Claim | Main supporting files |
|---|---|
| Regression target is the right substrate | `docs/challenge_alignment_notes.md`, `docs/track_a_metrics_notes.md`, final Phase 2 writeup |
| Data pipeline is leakage-aware | `src/qpitome_qrc/data/`, `scripts/prepare_phase2_spy_vix_dataset.py`, data validation notebook |
| ESN validates reservoir computing | `docs/classical_baseline_notes.md`, `notebooks/phase2_esn_regression_baseline_tuning.ipynb`, ESN result tables |
| QRC v0 was under-dispersed | `notebooks/phase2_tfim_qrc_light_touch_prototype.ipynb`, QRC v0 prediction exports |
| Leaky trajectory encoding improved QRC | `docs/may26_milestone6_design_probe_closeout.md`, leaky/final QRC notebooks and result tables |
| Regime-warning layer is transparent and useful | `docs/may27_milestone7_regime_transition_closeout.md`, forecast-aware regime notebooks |
| Neutral atoms are the primary Phase 3 hardware route | `docs/may28_milestone8_platform_phase3_plan.md`, final Phase 2 writeup |
| Phase 3 should remain simulator/emulator-first until access is confirmed | platform plan, current qBraid access uncertainty |

## 11. What Should Not Get Lost

1. The project became stronger when it moved from binary stress classification to continuous realized-volatility forecasting.
2. The ESN result is not a defeat; it is the diagnostic control that shows reservoir computing is relevant.
3. The QRC result is not random tuning; each major improvement addressed a specific weakness.
4. The central QRC bottleneck is tail-amplitude compression / under-dispersion.
5. The quantum advantage story must be mechanistic: many-body reservoir features, not vague speedup.
6. The Rydberg branch should start as a controlled experimental probe, not an overbuilt full forecasting system.

## 12. Immediate Next Step

Before deleting or restructuring files, create tests that freeze the current behavior:

- data split / leakage tests;
- metric tests;
- QRC statevector/feature-shape tests;
- prediction export smoke tests.

Only after those tests pass should Phase 2 documents, notebooks, and results be archived or cleaned for the Phase 3 branch.
