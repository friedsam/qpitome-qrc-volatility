# Track B destination-prediction protocol

Last updated: 2026-07-13

## Status

Track B is the primary scientific target. Current results are preliminary. Neither the classical reservoir nor the Rydberg reservoir has received a sufficient architecture study to support a positive or negative conclusion.

## Scientific question

> After a causal transition-warning episode begins, can the observed pre-episode market path predict whether the next eight weeks end in a positive or negative destination?

This question is distinct from volatility forecasting. Volatility may describe transition context, but it has not shown useful standalone destination-predictive power in the current audits.

## Episode admission

Admission is based on the causal restricted-HMM long-regime uncertainty signal:

- uncertainty threshold frozen from the early 60% of weekly history;
- threshold: `0.39936` in the first target audit;
- reset threshold: `0.35`;
- minimum gap between episodes: six weeks;
- one observation per uncertainty episode.

Admission is direction-neutral. The destination label is not defined from the future HMM state.

## Outcome

Primary provisional outcome:

- eight-week observed cumulative return;
- positive destination if return is greater than `+1%`;
- negative destination if return is less than `-1%`;
- outcomes inside `[-1%, +1%]` are excluded as neutral.

Current sample:

- 131 binary episodes;
- 85 positive and 46 negative;
- training: 83 episodes, 50 positive and 33 negative;
- holdout: 48 episodes, 35 positive and 13 negative.

## Evaluation

- chronological split date: `1999-12-17`;
- prequential evaluation on the holdout;
- before predicting episode `t`, training may include only episodes whose full eight-week outcome window has completed before episode `t`;
- preprocessing is frozen from pre-split weekly history;
- architecture selection must use training history only;
- the 48-episode holdout must not be used for architecture or threshold selection.

Primary metrics:

1. log loss;
2. Brier score;
3. ROC AUC;
4. balanced accuracy;
5. negative-destination recall;
6. positive-destination recall.

Plain accuracy is not a selection metric because holdout positive prevalence is `0.729`.

## Established baseline evidence

### Historical prior

- log loss: `0.61266`;
- Brier: `0.21041`;
- balanced accuracy: `0.50000`;
- negative recall: `0.00000`.

### 13-week momentum logistic model

- ROC AUC: `0.66374`;
- average precision: `0.84921`;
- log loss: `0.55108`;
- Brier: `0.18425`;
- balanced accuracy: `0.61538`;
- positive recall: `1.00000`;
- negative recall: `0.23077`.

Interpretation: momentum contains real ranking and probability information beyond the drifting class prior, but the `0.5` decision boundary is strongly biased toward the positive class. This is not a completed operational classifier.

### Compact directional logistic model

- ROC AUC: `0.61978`;
- log loss: `0.56299`;
- Brier: `0.19111`;
- balanced accuracy: `0.61099`;
- positive recall: `0.91429`;
- negative recall: `0.30769`.

It detects one additional negative case but has worse ranking and probability quality than momentum.

## Preliminary reservoir evidence

### One fixed ESN configuration

Configuration:

- 24 units;
- spectral radius `0.9`;
- leak `0.5`;
- input scale `0.35`;
- one random seed;
- final plus mean state readout;
- ridge penalty `10`.

Protected-offset result:

- ROC AUC: `0.55824`;
- log loss: `0.58725`;
- Brier: `0.20131`;
- positive recall: `0.94286`;
- negative recall: `0.15385`.

Conclusion: this configuration failed. It does not establish that ESNs are unsuitable for Track B.

### One fixed six-atom Rydberg configuration

Configuration:

- six-atom chain;
- exact statevector simulator;
- global Rabi drive;
- sequential detuning encoding;
- return plus uncertainty as successive control segments;
- final and temporal-mean occupations plus nearest-neighbor pair occupations;
- one fixed physical parameter set and readout penalty.

Protected return-plus-uncertainty result:

- ROC AUC: `0.68132`;
- average precision: `0.86424`;
- log loss: `0.55915`;
- Brier: `0.18915`;
- balanced accuracy: `0.59670`;
- positive recall: `0.88571`;
- negative recall: `0.30769`.

Paired diagnostic versus momentum:

- lower loss on 22 episodes, higher loss on 26;
- mean excess log loss: `+0.00807`;
- bootstrap 95% interval approximately `[-0.02756, +0.04379]`;
- mean probability shift on negative outcomes: `-0.00086`;
- negative cases were shifted downward only 46% of the time.

Conclusion: the configuration produced an exploratory AUC increase but no stable paired probability improvement. It is not a promoted model and does not establish that Rydberg dynamics are unsuitable.

## Required problem characterization before further claims

### Target structure

- inspect all negative episodes and their market contexts;
- quantify temporal clustering and era dependence;
- stratify by bear-side and bull-side origin;
- test whether the same destination concept is coherent across origins;
- assess modest horizon and neutral-zone perturbations without selecting on holdout performance;
- estimate uncertainty with event-block or era-aware resampling.

### Baseline envelope

- train-only threshold selection;
- calibration assessment;
- class-weighted or utility-weighted logistic comparator;
- one shallow nonlinear comparator;
- a fair ESN envelope selected only from pre-1999 rolling validation;
- multiple fixed seeds reported, not seed shopping.

### Reservoir diagnostics

Before predictive promotion, report:

- feature variance and saturation;
- effective rank and condition number;
- temporal-order sensitivity;
- memory of earlier weeks;
- interaction ablation;
- channel ablation;
- stability under modest physical-parameter perturbations;
- noise and finite-shot sensitivity;
- matched noninteracting and classical-reservoir controls.

## Architecture-selection rule

A small physically motivated architecture set may be compared using training-only rolling validation. Candidate differences must have a scientific interpretation, such as:

- return controlling detuning while uncertainty controls drive amplitude;
- spatially separated controls on atom subsets;
- chain versus compact two-dimensional interaction geometry;
- intermediate versus blockade-dominated interaction regimes;
- final-state versus sparse time-multiplexed observables.

Do not use the holdout to select among these designs.

## Promotion rule

A Rydberg model is promoted only if it shows a stable incremental result over the frozen classical comparator, preferably in log loss or Brier score, while preserving a defensible positive/negative recall tradeoff. An AUC-only improvement is exploratory unless calibration and paired behavior are also credible.

## Explicit non-conclusions

The following claims are not supported:

- that Track B has failed;
- that ESNs have failed generally;
- that Rydberg QRC has failed generally;
- that volatility forecasting should replace destination prediction;
- that the current `0.5` threshold is operationally appropriate;
- that the 48-episode holdout is large enough for aggressive architecture search.
