# Task-aligned QRC design for branch transition forecasting

## Status

Design note. No quantum advantage is claimed.

The generic ESN lane is closed for the current binary recovery-versus-relapse target. The next QRC experiment must not repeat the failed premise of feeding a broad generic path into a reservoir and asking the readout to discover the transition structure automatically.

The retained architecture is:

```text
frozen classical transition baseline
+
small reservoir correction
```

The redesign concerns the information presented to the reservoir and the observables extracted from it.

## Scientific target

At a causal stressed-but-stabilizing branch point, predict:

```text
recovery versus relapse
```

The existing `state_plus_motion` baseline already receives:

```text
stress_ratio
drawdown_120d
rv_ratio_5_20_branch
return_5d_branch
rv_5d_change_5d_branch
worst_return_5d_in_prior_window
```

Therefore the QRC should not spend scarce capacity reconstructing those same endpoint quantities.

The relevant question is:

> Among episodes with similar branch-point state and recent motion, does the ordering and interaction of the preceding path contain stable information about recovery versus relapse?

This is a conditional path-shape question.

## Design principle 1: preserve path ordering, not generic variance

PCA is not prohibited. It is simply not the validation criterion.

The failed compact-ESN experiment showed that 5 reservoir PCs retained roughly 95% of state variance and 10 PCs retained almost all variance, while incremental branch prediction worsened. High variance retention therefore did not identify complementary transition information.

For the QRC, the primary representation criterion is:

```text
Does the representation add stable OOS information beyond state_plus_motion?
```

not:

```text
How much marginal variance does it preserve?
```

## Design principle 2: do not make the QRC reproduce the easy component

The QRC should enter as a correction to the frozen classical baseline:

```text
baseline logit
+
small QRC correction
```

A zero correction must reproduce the baseline.

This avoids penalizing a constrained quantum model for failing to reconstruct information already handled well by the classical model.

## Candidate missing structure

The six-variable baseline describes branch-point state and a few recent summaries. It does not explicitly preserve temporal ordering within the preceding path.

Four transition-specific path contrasts are proposed for the first admission test.

### 1. Shock recurrence after apparent stabilization

Question:

> Did large negative shocks stop, or are they recurring late in the lookback window?

Two episodes can have the same worst prior decline and the same final 5-day return while differing in whether stress is concentrated early or repeatedly reappears near the branch point.

Candidate sequence:

```text
negative standardized return intensity
```

with daily values based on downside return magnitude normalized by contemporaneous volatility.

The informative object is ordering and recurrence, not the total downside magnitude already summarized by the baseline.

### 2. Rebound efficiency after damage

Question:

> Is positive price movement producing durable drawdown repair, or merely oscillation inside a damaged state?

Candidate sequence:

```text
positive standardized return
relative to concurrent drawdown repair
```

This distinguishes efficient recovery from high-variation rebound noise.

The baseline sees current drawdown and recent return but not how consistently positive movement repaired prior damage through the path.

### 3. Volatility-relaxation smoothness versus re-acceleration

Question:

> Is volatility deceleration persistent, or does the path repeatedly re-accelerate after short calm intervals?

Candidate sequence:

```text
change in log short/medium volatility ratio
```

The branch detector and baseline see the endpoint ratio and recent change. They do not preserve the sequence of relaxations and reversals leading to that endpoint.

### 4. Return-volatility phase relation

Question:

> Does volatility respond symmetrically to price movement, or do negative shocks continue to produce disproportionate volatility re-expansion?

Candidate paired sequence:

```text
signed standardized return
x
subsequent short-horizon volatility change
```

The target is lagged interaction and ordering. This is a plausible place for nonlinear temporal dynamics to matter because two paths can share nearly identical marginal summaries while differing in the sequence of shock and volatility response.

## Minimal encoded input

The first QRC should use a deliberately small input, not the previous generic six-channel path.

Preferred first candidate:

```text
u1(t): signed return / local volatility
u2(t): downside-shock recurrence signal
u3(t): change in log(RV5 / RV20)
u4(t): drawdown-repair increment
```

All channels must be:

- causal;
- computed only from information available by the branch date;
- clipped by fixed rules established before model evaluation;
- scaled without using test episodes;
- tested for redundancy with the six baseline variables.

The exact formulas should be implemented once in shared `src/` code and frozen before QRC comparison.

## Classical admission gate before quantum execution

The task-specific input must first pass a cheap diagnostic gate.

This is not a requirement that a classical model succeed before QRC can succeed. The purpose is to verify that the proposed channels actually represent the intended path contrasts and are not trivial duplicates of the baseline.

Required diagnostics:

1. **Endpoint redundancy audit**
   - regress or correlate each aggregate channel summary against `state_plus_motion`;
   - reject channels that are nearly deterministic rewrites of baseline features.

2. **Ordering-destruction control**
   - compare the true sequence with within-window temporal permutation or block permutation;
   - if a temporal model is unchanged, the proposed signal is not genuinely order-dependent.

3. **Matched low-complexity classical control**
   - use the exact same encoded sequence with a small classical temporal model;
   - this remains the closest control for any later QRC claim.

4. **Baseline-plus-correction evaluation**
   - all models are judged by incremental OOS value beyond `state_plus_motion`.

The admission gate does **not** use PCA retention or classical equivalence after PCA as evidence about quantum opportunity.

## First QRC architecture

The first circuit should be small enough to simulate exhaustively and later run on hardware without changing the scientific object.

Proposed geometry:

```text
4 qubits
one qubit per task-specific channel
40 sequential time steps
fixed recurrent entangling layer
no gradient training of the quantum reservoir
```

At each time step:

```text
encode four bounded channel values as local rotations
apply fixed entangling evolution
carry the quantum state forward
```

The circuit should expose a compact observable set rather than hundreds of features.

Initial observable budget:

```text
4 single-qubit Z expectations
4 nearest-neighbor ZZ expectations
```

Total:

```text
8 observables
```

This is intentionally smaller than the number of labeled episodes and far smaller than the failed high-dimensional ESN readouts.

A second observable family is justified only if the first family passes the admission test.

## Why this QRC could differ from the failed ESN

The claim is not that quantum dynamics are generically better than ESN dynamics.

The narrower hypothesis is:

> A compact entangling dynamical system may transform a small set of transition-specific ordered contrasts into stable interaction observables that add information beyond the classical branch-point baseline.

The possible value lies in:

- sequential interaction of multiple path contrasts;
- nonlinear phase-sensitive mixing;
- compact multi-channel observables;
- a readout dimension constrained before seeing performance.

If a matched classical reservoir using the same four channels and comparable readout dimension performs equally well, there is no quantum advantage claim.

## Fixed comparison chain

The first serious comparison should be:

```text
A. state_plus_motion

B. state_plus_motion
   + small classical temporal correction
   using the four task-specific channels

C. state_plus_motion
   + QRC correction
   using the same four task-specific channels
   and 8 fixed observables
```

Required ablation:

```text
D. QRC with temporal ordering destroyed
```

Optional only after success:

```text
E. one reduced-channel ablation identifying which interaction matters
F. noisy-simulator or hardware stability test
```

Do not begin with a grid over qubit count, depth, entanglers, encodings, observables, and seeds.

## Success criteria

The QRC proceeds only if it shows more than one isolated favorable metric.

Desired evidence:

- improvement over `state_plus_motion` in OOS probability quality and/or ranking;
- no catastrophic inversion between long-history and modern domains;
- improvement survives fixed reservoir seeds or fixed circuit instances;
- temporal-order ablation degrades the effect;
- the same effect is not fully reproduced by the matched compact classical reservoir;
- observable count remains small.

Given the tiny episode sample, no single AUC difference is sufficient.

## Falsifying result

The hypothesis is rejected if:

- task-specific channels remain redundant with the baseline;
- destroying temporal ordering does not matter;
- neither compact classical nor quantum temporal corrections add stable OOS information;
- QRC gains disappear under small circuit/noise perturbations;
- apparent gains require architecture search disproportionate to the sample size.

## Immediate implementation order

```text
1. implement causal task-specific path channels in src/
2. add exact unit tests for causality, clipping, and endpoint invariance
3. run redundancy and ordering-destruction diagnostics
4. freeze the four-channel sequence
5. build matched compact classical temporal control
6. only then implement the 4-qubit QRC
7. evaluate both as additive corrections to the same frozen baseline
```

This order prevents another round of generic reservoir architecture search and keeps the quantum experiment tied to the actual missing-information hypothesis.
