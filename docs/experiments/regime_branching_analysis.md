# Regime and branching-state analysis

## Purpose

This note records the exploratory analysis that led from broad volatility-regime inspection to a narrower, stakeholder-relevant forecasting problem.

The governing rule is:

> Let trivial baselines kill uninteresting problems. Spend advanced compute only on residual structure that is both real and useful.

The intended role of this analysis is therefore not to manufacture a difficult task for QRC. It is to identify which parts of the forecasting problem are already handled by cheap, trustworthy methods and isolate only the unresolved remainder.

## Data and guardrails

Exploratory dataset:

- `phase3_spy_vix_volatility_extended.csv`
- approximately 1993-04-27 through 2026-06-03
- SPY price, return, range, volume, realized-volatility, drawdown, and VIX-derived fields

Primary regime discovery was intentionally **VIX-blind** and **future-target-blind**:

- VIX columns were excluded from initial regime discovery.
- `future_rv_*` columns were excluded from regime discovery.
- Only causal contemporaneous and trailing market/realized-volatility information was used.

VIX was brought back afterward as an external comparator and triviality test.

This separation matters. If VIX is allowed to define the regimes, the analysis risks merely rediscovering low/medium/high VIX and then mistaking that for a novel regime taxonomy.

## Methodology wrapper

The exact exploratory code was not preserved and the thresholds were deliberately allowed to move while the problem definition was being refined. The analysis can nevertheless be reproduced from the following procedure.

### 1. Describe the curve without VIX

Inspect causal variables representing:

- realized-volatility level at multiple horizons;
- short/long volatility ratios and slopes;
- recent returns;
- drawdown;
- absolute-return and range behavior;
- recent reversals and large moves.

The first goal is descriptive: identify recurring fragments of market behavior without using the future outcome.

### 2. Segment into provisional recurring fragments

Use unsupervised grouping plus direct inspection of contiguous time fragments.

The first pass produced roughly six provisional fragment types:

- calm / low-activity behavior;
- mild or early stress;
- developing stress / drawdown;
- acute crisis;
- volatile rebound;
- easing / normalization.

These labels were treated as hypotheses, not final regimes.

### 3. Audit against historical episodes

Inspect major episodes with very different market histories, including examples such as:

- 1997-1998 stress episodes;
- dot-com decline;
- 2007-2009 crisis and aftermath;
- 2011 stress;
- 2015-2016 turbulence;
- 2018 volatility episodes;
- 2020 COVID shock;
- 2022 repeated deterioration/recovery cycles;
- 2024-2025 stress episodes.

Historical knowledge is used as an interpretive guide, not as the regime label itself.

The important test is whether similar curve morphologies recur across unrelated historical events.

### 4. Reintroduce VIX as a comparator

For each discovered fragment/state, compare its VIX distribution afterward.

This identifies which distinctions are trivial because VIX already captures them and which distinctions survive after matching on broad stress level.

### 5. Separate state from motion

A simple two-axis description of `stress level x direction` was tested and found incomplete.

At minimum, the data required separate descriptions of:

- current stress level;
- current damage/drawdown state;
- volatility rising or falling;
- market worsening or recovering;
- transition speed.

The same high stress and negative direction can correspond to slow deterioration or a rapid shock. These should not be collapsed.

### 6. Identify a recurrent branching state

The analysis then focused on a recurring high-stress state with the following broad morphology:

- realized volatility remains high;
- longer-horizon damage/drawdown remains visible;
- short-horizon volatility is below longer-horizon volatility or is decelerating;
- the recent decline has slowed or partially reversed.

This state appeared repeatedly across unrelated historical episodes.

It is best interpreted as an **unstable aftermath** or **branching state**:

> the market has stopped accelerating downward, but the eventual direction is not yet resolved.

### 7. Label future outcomes only after the state is defined

After extracting branching-state episodes using only information available at the time, classify future outcomes over a later horizon into:

- recovery;
- relapse / renewed deterioration;
- mixed or ambiguous.

This is the first point at which future information is used.

### 8. Apply the boring-baseline ladder

Evaluate progressively stronger cheap descriptions:

1. VIX alone;
2. simple contemporaneous state variables;
3. simple trailing-path summaries;
4. only then consider ESN/QRC or other expressive models.

The purpose is to eliminate easy structure before allocating advanced compute.

## Main findings

### 1. Broad volatility level is probably a trivial problem

VIX aligned strongly with the broad stress ordering discovered without VIX.

That supports a blunt conclusion:

> Much of the question `how stressed / how volatile is the market?` is already handled by VIX or equally cheap classical information.

Any advanced model that only reproduces this information has no meaningful advantage.

### 2. VIX does not fully describe dynamical position

The more interesting distinctions occur at similar stress levels.

Examples include:

- high volatility while still worsening;
- high volatility while rebounding;
- apparent stabilization followed by recovery;
- apparent stabilization followed by relapse.

In particular, worsening and volatile-rebound fragments can occupy similar realized-volatility ranges while having opposite price direction, drawdown evolution, and short/long volatility dynamics.

This suggests that VIX is a strong **stress gauge**, but not necessarily a complete **regime-transition descriptor**.

### 3. Crisis episodes do not share one temporal morphology

Historical crises showed qualitatively different paths:

- 2008-like episodes were sequential, path-dependent, and repeatedly alternated between worsening, rebound, renewed deterioration, and acute escalation.
- 2020 was much more compressed: calm -> rapid stress escalation -> acute crisis -> volatile recovery.
- 2022 looked more like repeated deterioration <-> partial recovery cycles than a single acute shock.

Therefore one universal temporal memory horizon or one universal crisis template is unlikely to be sufficient.

### 4. The unstable-aftermath state appears to be recurrent

A broad configuration of:

- high realized volatility;
- persistent longer-horizon damage;
- short-horizon deceleration;
- partial recent stabilization or rebound;

recurred across many unrelated periods.

This made it unlikely that the state was merely a clustering artifact from one famous crisis.

### 5. The unstable-aftermath state is genuinely branching

Across exploratory extractions, the state repeatedly resolved into all three outcome types:

- recovery;
- relapse;
- mixed/ambiguous.

The exact counts moved as thresholds were refined, so they should not be treated as final statistics. The robust result was the branching structure itself.

Earlier exploratory passes produced outcome splits such as approximately:

- 18 recovery / 11 relapse / 13 mixed;
- 29 recovery / 22 relapse / 15 mixed;
- 42 recovery / 28 relapse / 12 mixed.

These count changes are a reminder that the episode definition still needs to be locked before formal benchmarking.

### 6. Once the state is defined carefully, VIX does not separate recovery from relapse well

In a stricter exploratory comparison, recovery and relapse episodes could have very similar current VIX, realized-volatility level, drawdown, and short/long volatility ratio.

That is the key nontrivial residue:

> similar current stress, different future resolution.

### 7. Simple current-state features help, but do not obviously solve the branch

Exploratory boring classifiers using variables such as:

- realized-volatility level;
- short/long volatility ratio;
- drawdown;
- recent 5-day and 20-day returns;
- volatility slopes;
- VIX;

showed moderate but incomplete discrimination.

One exploratory walk-forward result was approximately AUC 0.66 for a simple current-state classifier.

This should be reproduced after the state and labels are locked. It is not yet a final benchmark.

### 8. VIX alone did not solve the branching task

One exploratory walk-forward pass produced approximately AUC 0.37 for VIX alone.

The exact value should not be overinterpreted before formal reimplementation, but the qualitative conclusion was stable: VIX was not a useful recovery-versus-relapse discriminator within the branching state.

### 9. Obvious hand-engineered 40-day path summaries did not add much

Tested summaries included variants of:

- prior volatility peak/minimum;
- volatility slope;
- peak/current ratio;
- cumulative return;
- absolute-return load;
- negative-day fraction;
- volatility reversals;
- return sign changes;
- local peak counts;
- largest prior move;
- path roughness;
- price-volatility coupling.

These did not produce a clean recovery-versus-relapse separation and often degraded exploratory cross-validation.

One exploratory comparison was roughly:

- current-state variables only: AUC ~0.61;
- path summaries only: AUC ~0.52;
- current + path summaries: AUC ~0.61.

A later exploratory implementation gave a stronger current-state result (~0.66), reinforcing that exact numbers are implementation-sensitive while the main conclusion remains: **naive path summaries do not close the gap**.

This negative result is important because it prevents a false claim that temporal information has already been exhausted. The summaries may simply destroy order and morphology.

## Current problem statement

A defensible current formulation is:

> Among markets that are already stressed but appear to be stabilizing, can we distinguish genuine recovery from renewed deterioration?

This problem is attractive because:

- it recurs across unrelated historical periods;
- VIX does not trivially solve it;
- simple contemporaneous classical features appear informative but incomplete;
- naive hand-engineered temporal summaries do not close the gap;
- the outcome is directly relevant to regime-transition forecasting.

The problem should be treated probabilistically, not as a deterministic classification task.

## Candidate resource-efficient pipeline

The emerging pipeline is modular rather than monolithic:

1. **Bulk volatility / broad stress**
   - VIX, HAR, or another cheap classical model.
   - Goal: remove the easy part of the task.

2. **Unstable branching states**
   - strong classical probabilistic classifier.
   - Goal: harvest as much remaining cheap signal as possible.

3. **Residual uncertainty**
   - ESN and/or QRC.
   - Goal: test whether richer nonlinear/dynamical representations add stable incremental value only where the classical primer remains uncertain.

Possible final forms include:

- HAR/VIX -> classical branch classifier -> ESN -> QRC;
- HAR/VIX -> classical branch classifier -> QRC;
- direct QRC, only if simpler stages do not exhaust the target.

The important design principle is division of labor. The reservoir does not need to rediscover information already captured by the primer.

## What remains to lock before advanced modeling

The broad exploration should stop once these items are formalized:

1. A causal, reproducible operational definition of the branching state.
2. A fixed future-outcome label definition for recovery, relapse, and mixed cases.
3. A walk-forward evaluation protocol at the episode level.
4. A strong boring probabilistic baseline with saved out-of-sample predictions.
5. A clear residual target for ESN/QRC.

Only after those items are locked should reservoir architecture be tuned against this problem.

## Tests performed during exploration

The following tests were attempted. Most should not be repeated casually unless the formal implementation changes the assumptions.

- VIX-blind unsupervised regime grouping.
- Historical episode inspection.
- VIX reintroduction as an external cross-check.
- Six-class regime interpretation.
- Simplification to `stress level x direction`.
- Failure analysis of the two-axis view.
- Separation of current state from transition motion.
- Search for a recurrent unstable-aftermath state.
- Repeated extraction under stricter/looser episode definitions.
- Recovery/relapse/mixed outcome labeling.
- Current-state comparison of recovery versus relapse.
- Matching on stress and damage.
- Prior-40-day peak/current and related path descriptors.
- Direct path-shape clustering.
- Counts of reversals, sign changes, and local peaks.
- Simple current-state classifier.
- VIX-only classifier.
- Simple path-summary classifier.
- Combined current-state + path-summary classifier.

## Important negative conclusions

These are worth remembering without bloating the main narrative:

- Six initial clusters should not be treated as six final regimes.
- `stress level x direction` is too simple; transition speed and volatility direction matter separately.
- `failed recovery` and `relapse` did not look like stable regimes by themselves; they were better interpreted as different outcomes from a branching state.
- VIX is not a good candidate for defining the regimes because it would trivialize broad stress segmentation.
- VIX is still essential as a comparator because any structure it already explains is not interesting QRC territory.
- Naive 40-day path summaries did not reveal a strong hidden temporal signal.
- Direct path-shape clustering did not produce a clean recovery/relapse partition.
- A difficult residual is useful only if it is stakeholder-relevant; difficulty alone is not sufficient.

## Reproducibility notes

The exploratory code was intentionally not preserved because the goal was problem discovery rather than benchmark production.

For formal reproduction:

- load the extended CSV;
- exclude all future columns from state discovery;
- initially exclude VIX from state discovery;
- define candidate branching episodes from only causal trailing variables;
- merge contiguous days into episodes rather than treating every day as independent;
- enforce minimum episode separation to avoid duplicate overlapping cases;
- classify future outcomes only after episode extraction;
- use walk-forward or purged walk-forward evaluation;
- save all episode-level out-of-sample predictions.

The first formal implementation should preserve episode metadata:

- start date;
- branch-point date;
- end date;
- historical event annotation if known;
- current-state variables;
- trailing path window;
- future outcome label;
- fold/training cutoff;
- baseline probabilities;
- ESN/QRC residual predictions.

## Figures worth producing later

1. Full-history regime timeline with VIX overlaid only after segmentation.
2. Representative curve fragments for calm, deterioration, acute escalation, unstable aftermath, and recovery.
3. Matched unstable-aftermath pairs with similar VIX/current state but opposite outcomes.
4. State-transition diagram showing calm, stress, crisis, unstable aftermath, recovery, and relapse.
5. VIX distributions across broad states versus overlapping VIX distributions inside the branching state.
6. Baseline probability calibration for recovery versus relapse.
7. Incremental performance of classical primer, ESN residual model, and QRC residual model.

## Working conclusion

The broadest volatility problem may be mostly uninteresting because VIX or similarly cheap classical information already captures much of it.

The more promising problem is narrower:

> identify and forecast the resolution of recurrent high-stress branching states that look similar in current stress level but diverge into recovery or renewed deterioration.

This problem has survived several cheap filters, but it has not yet been proven to contain quantum advantage.

That is the correct place to continue.
