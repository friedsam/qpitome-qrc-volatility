# Phase 3 continuation handoff

qBraid / MITRE / JonesTrading — post-D1 failure analysis, weekly-HMM redesign, and preliminary Track B rebuild

Date: 2026-07-13

## 1. Purpose

This handoff supersedes the 2026-07-10 post-D1-lock handoff. The earlier handoff correctly emphasized disciplined evaluation, repository hygiene, and the need to give ESN and Rydberg/QRC a fair chance. Its central scientific claim, however, no longer stands: D1 is not a valid locked predictive result because most of its apparent signal is reproduced by construction-implied first-passage geometry.

The project has therefore undergone a fundamental redesign. The primary target is now weekly transition-context destination prediction (Track B), with onset detection (Track A) retained as secondary work.

## 2. Fundamental correction: the D1 result was invalidated

### What D1 claimed

D1 used four day-5 first-passage features:

- endpoint return;
- recovery distance;
- relapse distance;
- barrier width.

The post-1990 confirmatory evaluation appeared strong:

- 299 predictions in 150 clusters;
- D1 log loss `0.6146`;
- D1 ROC AUC `0.7194`;
- prior log loss `0.6871`.

### What the exact audit found

The four D1 features are nearly algebraically redundant:

- barrier width equals recovery distance plus relapse distance;
- after standardization the D1 design matrix has exact rank two;
- the feature set reduces largely to endpoint position and scale.

A zero-parameter first-passage corridor-position null reproduced or exceeded the apparent D1 performance:

- calibrated lambda log loss: `0.6125`;
- D1 log loss: `0.6146`;
- zero-parameter first-passage null log loss: `0.6185`;
- prior log loss: `0.6871`;
- null ROC AUC: `0.7257`;
- D1 ROC AUC: `0.7194`;
- correlation between null and D1 predictions: `0.9442`;
- approximately `94.6%` of D1's log-loss improvement over the prior was reproduced by the null.

### Interpretation

This was not necessarily conventional future leakage. It was a more subtle construction problem: the target and features shared first-passage corridor geometry. The model mostly recovered where the episode already sat between fixed barriers.

### Permanent decision

- Retire D1 as the project headline and as the locked comparator.
- Retain the old barrier framework only as an event-detection or historical-analysis tool.
- Do not return to barrier-defined destination prediction for the current subproblem.

## 3. Literature work and the new conceptual basis

A close reading of Maheu, McCurdy, and Song motivated a four-state weekly regime framework:

1. bear;
2. bear rally;
3. bull correction;
4. bull.

The restricted transition topology separates same-sign returns occurring inside different long regimes. The original paper uses Bayesian posterior simulation and predictive densities. The repository implementation is a transparent masked Baum-Welch EM approximation with sign-clamped state means, not a faithful Bayesian replication.

The literature review supported three design decisions:

- four states can be materially more informative than a two-state bear/bull model;
- the exact restricted topology may aid semantic identification and extreme-event robustness, but should be compared against unrestricted HMM4;
- filtered state probabilities can define causal transition context without using smoothed future information.

Financial QRC literature supports dense sequential time-series processing, especially for realized-volatility forecasting. It does not directly establish that a Rydberg reservoir will solve rare conditional destination prediction. The project must therefore build its own matched controls and mechanism tests.

## 4. Weekly HMM baseline ladder

### Protocol

- weekly S&P 500 returns;
- 520-week initial history;
- expanding causal evaluation;
- annual parameter refits;
- filtered states only;
- one-step predictive density;
- 3,471 forecast weeks.

### Predictive-density results

| Model | Mean LPD | Total LPD |
|---|---:|---:|
| Restricted HMM4 | -2.07638 | -7207.12 |
| Unrestricted HMM4 | -2.07859 | -7214.77 |
| HMM2 | -2.09757 | -7280.66 |
| IID Gaussian | -2.21780 | -7697.98 |

Differences:

- restricted HMM4 minus unrestricted HMM4: `+7.6538` total;
- restricted HMM4 minus HMM2: `+73.5465`;
- restricted HMM4 minus IID Gaussian: `+490.8616`.

### Interpretation

- Four states materially outperform two.
- The exact restricted topology contributes only a small average gain.
- The restricted advantage is concentrated in extreme weeks and crisis periods, especially COVID.
- Unrestricted HMM4 remains the neutral predictive comparator.
- Restricted HMM4 remains useful for interpretable state identity and tail behavior.

### Restricted-state profiles

- state 0: high-volatility stress/bear;
- state 1: positive bear rally;
- state 2: negative bull correction;
- state 3: low-volatility bull.

The aggregate profiles support the intended semantics. Parameter identity stability across annual refits has not yet been fully audited because the original long run lost fit history when manifest serialization failed. A checkpointed rerun remains desirable but is not the immediate Track B bottleneck.

## 5. Track A: transition-onset detection

### Exploratory label

- broad bear side: states `{0,1}`;
- broad bull side: states `{2,3}`;
- positive if the dominant broad regime flips within four weeks and persists for at least three of the next four weeks.

This is a retrospective warning-window label. Inputs remain causal. It is a proxy for imminent persistent regime transition, not literal observation of counterfactual branching potential.

### Triviality audit

Holdout prevalence: `0.14967`.

Best simple diagnostics:

- long-regime uncertainty: AP `0.3357`, AUC `0.7483`;
- one-week probability motion: AP `0.2870`, AUC `0.7312`;
- realized-volatility measures were only slightly above no-skill prevalence.

This established that onset is not simply high volatility.

### Event-time trajectory

Using frozen training thresholds and 58 holdout events:

- uncertainty warned on 45/58 events (`77.6%`) in weeks `-4` to `-1`;
- median first warning: three weeks before transition;
- probability motion warned on 50/58 events (`86.2%`), but was noisier and spiked mainly at week zero;
- non-event false-alarm rates after excluding ±8-week event neighborhoods were approximately `6.3%` and `7.0%`.

Uncertainty rose from a median around `0.446` at week `-4` to `0.816` at week `-1`.

### Status

Track A is a credible work-in-progress and fallback result. It should not replace Track B merely because it is easier. Its false-alarm estimate is conditional on event-neighborhood exclusion and must not be described as a general weekly false-positive rate.

## 6. Track B: destination prediction

### Current target

Admission:

- first causal crossing into elevated restricted-HMM long-regime uncertainty;
- threshold frozen from early history;
- reset below `0.35`;
- minimum six-week episode gap;
- one sample per episode.

Outcome:

- observed eight-week cumulative return;
- positive if above `+1%`;
- negative if below `-1%`;
- neutral zone excluded.

Evaluation:

- chronological split at `1999-12-17`;
- prequential holdout;
- training includes only episodes whose outcome windows have completed before the prediction date.

### Sample

- 131 binary episodes;
- 85 positive, 46 negative;
- training: 83, with 50 positive and 33 negative;
- holdout: 48, with 35 positive and 13 negative.

### Class drift

Positive prevalence rises from `0.602` in training to `0.729` in holdout. Therefore:

- always-positive accuracy is `72.9%`;
- plain accuracy is not a valid selection metric;
- log loss, Brier score, AUC, balanced accuracy, and class-specific recall are required.

The drift is partly compositional:

- bear-side holdout prevalence: `0.781`;
- bull-side holdout prevalence: `0.625`.

The target also exhibits temporal clustering, including dense episode groups in 1973-75 and 1981-82.

## 7. Track B classical baselines

### Historical prior

- log loss `0.61266`;
- Brier `0.21041`;
- balanced accuracy `0.500`;
- negative recall `0.000`.

### 13-week momentum logistic model

- AUC `0.66374`;
- AP `0.84921`;
- log loss `0.55108`;
- Brier `0.18425`;
- balanced accuracy `0.61538`;
- positive recall `1.000`;
- negative recall `0.23077`.

This is currently the strongest simple probability baseline. It has a major operational weakness at threshold `0.5`: it predicts nearly everything positive. The ranking signal is real, but the classifier is not finished.

### Compact directional logistic model

- AUC `0.61978`;
- log loss `0.56299`;
- Brier `0.19111`;
- balanced accuracy `0.61099`;
- positive recall `0.91429`;
- negative recall `0.30769`.

It improves negative recall but worsens ranking and probability quality.

### Important threshold finding

On the holdout, raising the momentum threshold to `0.55` produced balanced accuracy around `0.664`, positive recall `0.943`, and negative recall `0.385`. This was a diagnostic, not a valid operating-point selection because the threshold was inspected on holdout outcomes. Future threshold selection must use training-only utility or validation.

## 8. Shared reservoir-path dataset

A model-agnostic causal tensor was built:

- 131 episodes;
- 13 weeks per episode;
- 8 channels;
- shape `(131, 13, 8)`.

Channels:

- weekly return;
- 4-week and 13-week cumulative return;
- 4-week and 13-week realized volatility;
- bull-side HMM probability;
- long-regime uncertainty;
- one-week probability motion.

Each path ends at the episode date. Scaling is fixed from pre-split weekly history. The same paths are intended for matched classical and Rydberg reservoirs.

## 9. Preliminary ESN test

Only one ESN configuration was tried:

- 24 units;
- spectral radius `0.9`;
- leak `0.5`;
- input scale `0.35`;
- one seed;
- final plus mean state features;
- ridge penalty `10`.

Protected momentum-plus-ESN result:

- AUC `0.55824`;
- log loss `0.58725`;
- Brier `0.20131`;
- positive recall `0.94286`;
- negative recall `0.15385`.

This configuration failed. The test was too superficial to justify rejecting ESNs. A fair classical reservoir envelope still requires a compact train-only architecture search, multiple fixed seeds, and reservoir diagnostics.

## 10. Preliminary Rydberg work

### Input encodings

Hardware-friendly bounded encodings were created:

- return only;
- return plus uncertainty;
- return plus volatility as an ablation.

Return values used the available range without excessive saturation. Uncertainty did not saturate. Volatility saturated more often and remains an ablation, not a new primary target.

### First exact-statevector architecture

- six-atom chain;
- 64-dimensional exact statevector;
- global Rabi drive;
- nearest-neighbor interaction approximation;
- return and uncertainty applied sequentially through the same detuning control;
- fixed physical parameters;
- final and temporal-mean occupations plus nearest-neighbor pair occupations;
- protected momentum offset.

### Results

Return-only protected model:

- AUC `0.61319`;
- log loss `0.57749`;
- Brier `0.19636`;
- negative recall `0.15385`.

Return-plus-uncertainty protected model:

- AUC `0.68132`;
- AP `0.86424`;
- log loss `0.55915`;
- Brier `0.18915`;
- positive recall `0.88571`;
- negative recall `0.30769`.

### Paired diagnostic

Compared with momentum on the same 48 dates:

- lower Rydberg loss on 22 episodes;
- higher Rydberg loss on 26;
- mean Rydberg-minus-momentum log loss `+0.00807`;
- bootstrap interval approximately `[-0.02756, +0.04379]`;
- negative-outcome mean probability shift `-0.00086`;
- only 46% of negative cases shifted downward.

### Correct interpretation

The first Rydberg configuration showed an exploratory ranking increase but no stable paired probability gain. It is not a promoted model. It is also far too preliminary to support a negative conclusion about Rydberg QRC or Track B.

The sequential same-control encoding is physically weak because it treats uncertainty as another return-like detuning pulse. Alternative physically motivated mappings remain open, including simultaneous dual control and spatially separated controls.

## 11. Methodological failures during the latest work

The latest chat introduced several process failures that the next model must not repeat:

1. **Premature closure after one configuration.** One ESN and one Rydberg setting were treated as though they represented their model families.
2. **Prompt-by-prompt reaction.** Recommendations changed too quickly after each output instead of following a stable research program.
3. **Weak baseline maturity.** The momentum model's threshold pathology was recognized, but a full baseline envelope and train-only operating-point selection were not completed before reservoir judgment.
4. **Insufficient problem characterization.** Negative episodes, target robustness, temporal clustering, era dependence, and origin-regime heterogeneity remain incompletely studied.
5. **No reservoir-dynamics workup.** Feature rank, variance, memory, temporal-order sensitivity, interaction dependence, and parameter stability were not characterized before interpreting prediction.
6. **Overstated stop rules.** Track B was nearly abandoned after less than a serious architecture study.
7. **Volatility drift.** Volatility/GARCH was repeatedly suggested despite the user's explicit and empirically supported position that volatility has no useful destination-predictive power here. Volatility may remain contextual or an ablation, not the primary target.

## 12. What is established, provisional, and unresolved

### Established

- D1's headline predictive claim is artifact-dominated and retired.
- Four-state weekly HMMs outperform HMM2 and IID Gaussian in causal predictive density.
- Restricted HMM4 has interpretable states and modest extreme-tail advantages over unrestricted HMM4.
- HMM uncertainty gives a meaningful early-warning trajectory for persistent broad-regime transitions.
- Track B is not solved by class prevalence alone.
- 13-week momentum contains real destination-ranking signal.
- One ESN configuration failed.
- One serial-detuning Rydberg configuration produced an exploratory AUC increase but no stable probability improvement.

### Provisional

- Eight-week return with a ±1% neutral zone is the preferred Track B target.
- Momentum is the current primary simple baseline.
- Return plus uncertainty is more promising than return alone for Rydberg encoding.

### Unresolved

- Whether negative destinations form one coherent class.
- Whether target performance is stable across eras, origins, and modest target perturbations.
- How to select a threshold or utility function without holdout contamination.
- What a fair ESN performance envelope is.
- Whether current reservoir features possess useful memory, rank, and nonlinear interaction structure.
- Which physically motivated Rydberg architecture is appropriate.
- Whether any Rydberg advantage survives interaction, order, noise, and matched-classical ablations.

## 13. Required next phase

Do not run another isolated classifier first.

### A. Characterize the target

- produce a dated table of all negative episodes;
- inspect market context and origin regime;
- quantify clustering and era composition;
- test bounded horizon and neutral-zone perturbations using predeclared robustness analysis;
- decide whether positive/negative destination is coherent across bear-side and bull-side origins.

### B. Mature the baseline envelope

- calibrate and select thresholds using training-only rolling validation;
- add a class-weighted or utility-weighted logistic comparator;
- add one shallow nonlinear comparator;
- build a compact but fair ESN architecture grid using training-only validation and multiple fixed seeds;
- preserve the 48-episode holdout.

### C. Diagnose reservoirs before classification

For ESN and Rydberg candidates, measure:

- feature variance;
- effective rank;
- condition number;
- memory of earlier time steps;
- temporal-order sensitivity;
- channel sensitivity;
- interaction ablation;
- stability to modest parameter changes;
- finite-shot and noise sensitivity for the Rydberg model.

### D. Select Rydberg architectures scientifically

Use a small predeclared training-only set, for example:

- return-to-detuning plus uncertainty-to-Rabi simultaneous control;
- spatially separated return and uncertainty controls;
- chain versus compact 2D geometry;
- intermediate versus blockade-dominated interactions;
- final-state versus sparse time-multiplexed observables.

Do not select on the holdout.

## 14. Repository organization

The flat `scripts/modeling` directory is being replaced with:

```text
scripts/modeling/day5_barrier/
scripts/modeling/weekly_regimes/
scripts/modeling/task_a_onset/
scripts/modeling/task_b_destination/
scripts/modeling/classical_models/
scripts/modeling/quantum_models/
scripts/modeling/legacy_unclassified/
```

The safe reorganization tool is:

```text
scripts/maintenance/reorganize_modeling_tree.py
```

Run dry first, then apply:

```bash
python scripts/maintenance/reorganize_modeling_tree.py
python scripts/maintenance/reorganize_modeling_tree.py --apply
```

It uses `git mv`, updates Markdown references, and writes a layout audit and move manifest.

Documentation index:

```text
docs/README.md
```

Current Track B protocol:

```text
docs/protocols/track_b_destination_protocol.md
```

## 15. Current recent commits

Recent scientific sequence before repository cleanup:

```text
d784f8d  Add paired diagnostic for Task B Rydberg increment
25d2fe4  Add first exact-statevector Task B Rydberg reservoir test
d8f1b53  Prepare hardware-friendly Task B Rydberg input encodings
3be59b2  Add matched ESN baseline for Task B reservoir paths
a250276  Prepare causal Task B episode paths for reservoir models
1438a0e  Add leakage-safe Task B classical baseline ladder
8d73f20  Add Task B class-drift and origin-regime audit
08cacf4  Add first Task B destination target audit
ead8788  Add Task A event-time warning trajectory
```

Documentation/cleanup commits added afterward should be read from `git log`.

## 16. Working principles for the next chat

- Track B remains primary.
- Do not pivot to generic volatility forecasting.
- Track A is a secondary result and fallback, not the default escape route.
- Do not declare a model family failed after one setting.
- Do not architecture-shop on the holdout.
- Do not confuse honest reporting with premature closure.
- Maintain a stable experimental program across turns.
- The goal is to make Track B work if it can be made to work under a credible protocol, and to understand exactly why if it cannot.
