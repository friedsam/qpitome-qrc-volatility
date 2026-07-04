# Phase 3 Research History: What We Tried, What Failed, and What We Learned

## Overview

Phase 3 work began on June 28, 2026. This document captures the research path through July 4, 2026: the hypotheses tested, the branches explored, the failures that redirected the project, and the evidence that led to the current true-temporal Rydberg strategy.

The goal is not to present a polished final narrative. It is to preserve the actual research logic so that later submission work can distinguish genuine progress from hindsight reconstruction.

Team structure during this period:

- Claudia Friedsam: project lead, scientific direction, implementation decisions, experiment execution, interpretation, and validation.
- ChatGPT: primary reasoning partner, experiment design, adversarial review, repo analysis, interpretation, and strategy.
- Fable: secondary implementation and critique helper used when additional execution capacity or an independent view was useful.

The core working principle evolved into:

> First exploit a result. Then try to kill it.

The project objective remained unchanged:

> Find a genuine, defensible quantum advantage under realistic cost and deployment constraints.

---

## 1. Starting point: strong classical forecasting, weaker quantum forecasting

Phase 2 ended with a clear asymmetry.

### Classical baseline picture

For the original fixed chronological split and 20-day realized-volatility target:

- HAR ridge: test RMSE about 0.1007, QLIKE about -2.079, MZ R2 about 0.354.
- Persistence was materially weaker.
- The strongest ESN exports reached test RMSE around 0.0718-0.0719 and MZ R2 around 0.53-0.54.
- A separate five-seed ESN confirmation was weaker, which means there were already multiple legitimate ESN reference levels rather than one universal number.

This established a high classical bar.

### Phase 2 TFIM/QRC picture

The Phase 2 quantum branch improved substantially over its early versions but remained below ESN on continuous forecasting.

Representative final metrics:

- RMSE about 0.0951
- QLIKE about -2.230
- MZ R2 about 0.189
- correlation about 0.435

A feedback-TFIM reference was similar but slightly weaker on the final test split.

The main lesson from Phase 2 was therefore straightforward:

> A quantum reservoir could be made meaningfully predictive, but it was not competitive with the strongest ESN on ordinary continuous volatility forecasting.

That left two possible directions:

1. keep improving continuous regression until the quantum model catches ESN;
2. ask whether the quantum reservoir has a different capability that RMSE and QLIKE do not reveal.

Phase 3 initially explored both.

---

## 2. Early Phase 3: the first clue that the metric was hiding model capacity

The first major Phase 3 branch used an RF-QRC ring architecture.

On ordinary continuous forecasting, it was not competitive:

- worse RMSE than the Phase 2 QRC;
- much worse QLIKE than ESN;
- lower forecast amplitude than the classical reference.

A conventional regression-first reading would have killed the branch.

But the tail behavior looked different.

Historical fixed-split results showed:

- Phase 2 QRC q95 amplitude F1: 0
- RF-QRC ring q95 amplitude F1: about 0.414
- ESN reference q95 amplitude F1: about 0.575

The quantum model was still not globally best, but the qualitative change was too large to ignore.

A forecast-aware regime layer made the contrast sharper:

- RF-QRC q95 crisis-like F1 about 0.430
- historical ESN reference about 0.360
- Phase 2 QRC about 0.196

This was not a valid quantum-advantage claim because the protocol was fixed-split and not yet comparable to the later purged walk-forward setup.

But it produced the most important early Phase 3 pivot:

> A model that looks poor under regression metrics may still contain useful rare-event ranking information.

That observation changed the search target from pure level forecasting to rare-tail discrimination.

---

## 3. RF-QRC tuning: useful signals, unstable gains

Several branches tried to determine whether the RF-QRC tail behavior could be strengthened into a robust result.

### Leak and ridge sweep

Leak and ridge regularization strongly changed:

- prediction amplitude;
- q90/q95 F1;
- precision-recall trade-offs;
- crisis-call frequency.

The best crisis-like F1 appeared near moderate leak and ridge values, but continuous forecast quality remained weak.

The key lesson was negative but important:

> The tail result was highly sensitive to readout regularization and temporal integration settings.

That meant any future claim would need strict validation-only selection and a much cleaner protocol.

---

## 4. RF-QRC architecture probes

### Second encoding and ring entanglement

A tail-probe study compared increasingly structured variants.

The ring architecture improved correlation and tail F1 relative to simpler variants.

This supported the idea that reservoir geometry mattered.

### Structured level/rate entanglement

More elaborate entangling structures were then tested:

- cross-matched
- cross-all
- block-plus-cross

None beat the ring control under the experiment's own gate.

The ring remained best.

Lesson:

> More structure was not automatically better. The simpler ring was more useful than several hand-designed alternatives.

This was an early sign that architectural complexity had to be justified by evidence, not intuition.

---

## 5. Time multiplexing: more rank did not mean more value

Time multiplexing increased the effective feature rank of the reservoir.

Some q95 F1 values improved, especially with more virtual nodes.

However:

- no tested configuration passed the predefined time-multiplexing gate;
- higher effective rank did not translate into a sufficiently robust performance gain;
- the extra complexity was not justified.

This became a useful negative result:

> Increasing representational dimension alone did not solve the problem.

The project needed qualitatively different temporal capacity, not merely more features.

---

## 6. One reservoir, two heads: direct tail classification helped, but validation failed at q95

A separate branch tested one shared RF-QRC representation with:

- a ridge regression head for continuous volatility;
- logistic heads for q80, q90, and q95 events.

The direct classifiers improved q80/q90 ranking behavior relative to amplitude-thresholding the regression output.

But the q95 validation set was degenerate.

The result was a pathological q95 classifier that effectively called almost everything positive on test.

Lesson:

> Direct tail heads may be useful, but q95 model selection is impossible when the validation period contains no positive events.

This failure later became one reason to treat valid-fold policy as a first-class methodological issue.

---

## 7. Qubit scaling: bigger was not better

Two scaling studies produced a consistent message.

### Qubit-scaling transition classifier

Historical fixed-split results showed:

- HAR q95 AP about 0.399
- 8-qubit QRC alone about 0.284
- 8-qubit HAR+QRC about 0.423

The 8-qubit hybrid modestly improved over HAR, while larger QRC-only variants did not improve monotonically.

At the same time, feature-generation time increased sharply with qubit count.

This produced an important early clue:

> Quantum features may be more useful as conditional augmentation than as a universal replacement for strong classical structure.

That idea would later reappear in discussions of selective quantum invocation and ROI.

### TFIM N/H scaling

The TFIM scaling study pushed qubit count and field strength.

Results:

- effective rank increased;
- runtime increased dramatically;
- tail F1 remained mostly near zero;
- 12-qubit simulations took thousands of seconds without proportional forecast improvement.

Lesson:

> Larger Hilbert spaces did not create useful value by themselves.

This was a strong negative cost-performance result and helped end the idea that simple TFIM scaling might rescue the architecture.

---

## 8. First Rydberg temporal sweep: generic temporal processing was not enough

A large early Rydberg sweep tested:

- two scalar inputs;
- lookbacks 10, 20, and 40;
- three detuning windows;
- 18 total configurations.

The models compared:

- raw baseline;
- Rydberg temporal features;
- raw plus Rydberg temporal features.

Aggregate test results showed:

- q90 AP improved in only 2/18 configurations;
- q95 AP improved in only 2/18;
- q95 AUC improved in 12/18;
- RMSE improved in 5/18.

This was a subtle result.

The reservoir often changed ranking geometry enough to improve AUC, but usually did not improve precision-recall performance.

Lesson:

> Generic Rydberg temporal processing was not sufficient. The model could rearrange ranking structure without creating enough useful rare-event concentration.

This branch was therefore a partial failure, not the origin of the current advantage.

That distinction is critical because later true-temporal Rydberg results should not be merged with this earlier sweep.

---

## 9. The important pivot: true temporal dual-chain Rydberg dynamics

The later Phase 3 architecture changed the problem more fundamentally.

Instead of static or weakly temporal feature expansion, one quantum state evolved through the full input window.

Core structure:

- exact-state simulation;
- one state propagated through all temporal anchors;
- input level mapped to detuning;
- input rate mapped to Rabi amplitude;
- dual-chain slow/fast geometry;
- site occupations and pair correlations as readout features.

A base purged walk-forward comparison evaluated:

- raw baseline;
- raw products;
- memoryless Rydberg control;
- shuffled-anchor control;
- true temporal Rydberg model.

Median q95 AP values were approximately:

- raw: 0.087
- raw products: 0.091
- memoryless: 0.144
- temporal: 0.212

This was the first result that looked structurally different from the failed earlier branches.

The pattern suggested a capacity ladder:

1. raw state was weak at deep-tail ranking;
2. trivial pair products added almost nothing;
3. nonlinear reservoir transformation helped;
4. true temporal dynamics helped more;
5. multi-timescale temporal processing might help further.

The result did not prove quantum advantage.

But it created a legitimate opening.

---

## 10. Mechanism diagnostics: temporal capacity was measurable

The next question was whether the true-temporal model was merely another nonlinear feature map.

A temporal-memory diagnostic tested controlled synthetic targets.

Key idea:

A memoryless additive model can learn same-time nonlinear functions, but it cannot represent cross-time products such as:

u[a] * u[b]

with a linear readout unless those interactions are already explicitly included.

The true temporal Rydberg model showed positive cross-time capacity that decayed with temporal separation.

The memoryless model did not.

This established something much stronger than an ordinary forecast metric:

> The temporal reservoir possessed a measurable cross-time nonlinear capacity absent from the memoryless control.

The dual-chain diagnostic also showed a large difference between slow-chain and fast-chain connected correlations, supporting the interpretation that the geometry created multiple internal timescales.

This did not yet prove that those capacities caused the market-forecasting improvement.

But it supplied a plausible mechanism that could be independently tested.

---

## 11. Fold-level analysis changed the interpretation again

The q95 results were then examined fold by fold.

Valid q95 folds were 1, 2, 4, and 5. Fold 3 had degenerate q95 test labels.

The raw versus temporal AP pattern was approximately:

| Fold | Raw AP | Temporal AP |
|---|---:|---:|
| 1 | 0.534 | 0.184 |
| 2 | 0.096 | 0.103 |
| 4 | 0.078 | 0.099 |
| 5 | 0.016 | 0.027 |

The average alone hid the actual story.

### Fold 1

Fold 1 had the least training data in the expanding walk-forward setup.

It also covered a period in which major future volatility events were highly visible in engineered classical stress variables.

The raw baseline therefore had two advantages:

- lower statistical complexity under the smallest training set;
- highly informative direct classical features.

The quantum reservoir likely diluted an already strong signal.

### Later folds

In folds 2, 4, and 5, raw q95 AP was much weaker.

The temporal model improved over raw in all three.

This created a more useful working hypothesis:

> Quantum temporal processing may be most valuable when current-state classical separability is weak and enough training history exists to fit the higher-capacity representation.

This shifted the project away from one global leaderboard number toward a conditional-value question:

> Under what market conditions does the quantum model add information that cheap classical models do not already have?

That is now central to both scientific interpretation and stakeholder ROI.

---

## 12. Multi-lookback Rydberg: the strongest current opening

The next branch combined multiple temporal scales.

A multi-lookback ridge strategy used both long and short temporal reservoirs.

Under the currently selected valid-fold protocol, the strongest exact q95 AP reached about:

- exact: 0.3168
- exact-train to 1000-shot noisy-test: 0.2696
- 1000-shot noisy-train to noisy-test: 0.2551

This suggested:

- multi-timescale temporal processing may add substantial deep-tail value;
- finite-shot sampling degrades the result but does not erase it.

However, the current result is not yet canonical because:

- fold comparability must be standardized;
- full Track A metrics are missing;
- the Phase 3 ESN comparison under the exact same evaluator must be recovered or rerun;
- search breadth and model-selection risk must be documented.

So this is the strongest opening, not yet the final claim.

---

## 13. Cost became a first-class scientific variable

The challenge is not simply to maximize AP.

Quantum cost matters because:

- purged walk-forward multiplies reservoir evaluation work;
- multi-lookback uses more than one reservoir task per prediction;
- finite-shot sampling increases runtime and hardware credits;
- hardware-native anchor extraction may require repeated execution depending on implementation.

The relevant question is therefore not:

> Does the quantum model win somewhere?

It is:

> Does the incremental predictive value justify the incremental quantum cost under conditions where classical models are weak?

This led to the idea of selective quantum invocation:

- run cheap classical models continuously;
- detect regimes where their separability or confidence is poor;
- invoke the quantum reservoir only where it has evidence of incremental value.

This may ultimately be a more defensible practical advantage than daily unconditional QPU use.

---

## 14. What the project learned in one week

Between June 28 and July 4, the project moved through several distinct research phases:

1. confirmed that continuous regression remained classically dominated;
2. found that tail behavior could differ sharply from regression quality;
3. tested and rejected several simple routes to improvement;
4. showed that more features, more qubits, and more hand-designed structure were not sufficient;
5. learned that generic Rydberg temporal processing mostly changed AUC rather than AP;
6. developed a true-temporal dual-chain reservoir;
7. found a q95 ranking signal not reproduced by raw pair products;
8. demonstrated measurable cross-time nonlinear capacity;
9. discovered that fold-level market structure mattered more than one average;
10. identified multi-timescale temporal processing as the strongest current candidate;
11. quantified finite-shot sampling degradation;
12. reframed the final value proposition around conditional market regimes and ROI.

This is substantial progress for a six-day period.

The project did not arrive at one lucky configuration and stop.

It created a sequence of falsifiable hypotheses, killed several of them, and used the failures to narrow the search.

That is the strongest part of the research history.

---

## 15. Current open questions

The main unresolved questions are now much narrower than they were at the start of Phase 3.

### Scientific

- Does the multi-lookback q95 gain survive a fully canonical fold policy?
- Does the current ESN match or beat it under exactly the same evaluator?
- Is the gain tied to low raw separability, larger training history, temporal nonstationarity, or another market descriptor?
- Does true temporal ordering remain necessary after multi-lookback expansion?
- Can the mechanism diagnostic be linked quantitatively to forecasting gain?

### Statistical

- How sensitive is the result to fold composition?
- How much of the gain survives pooled out-of-fold evaluation?
- How much multiple-comparison risk was introduced by the AP-push search?
- What validation-only selection rule is defensible for q95 when some folds are degenerate?

### Hardware

- How much device noise is added beyond finite-shot sampling noise?
- Do the selected positive and negative hardware windows preserve simulator ordering?
- Which observables are most hardware-stable?

### Economic

- What is the true quantum execution cost per prediction?
- Can selective invocation preserve most of the gain at much lower cost?
- In which market regimes would the incremental warning value justify the QPU expense?

---

## 16. Research principle going forward

The next stage should not maximize the number of experiments.

It should maximize the probability of resolving the remaining claim.

For every new experiment:

1. state the hypothesis;
2. state what result would kill it;
3. compare under the same protocol;
4. record fold-level behavior;
5. record incremental cost;
6. only promote results that survive adversarial checks.

The project now has enough history to avoid repeating failed branches.

The objective for the remaining period is not to make the repository look busier.

It is to determine whether the strongest surviving result is a genuine conditional quantum advantage, and if so, to characterize precisely where the value can be harvested.
