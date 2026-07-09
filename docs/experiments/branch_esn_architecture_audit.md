# Branch ESN architecture audit

## Purpose

This note freezes the findings that justify a bounded ESN repair before any further model interpretation.

The primary task remains direct forecasting of `recovery` versus `relapse` from the branching state. The current binary target is retained for the one-day ESN audit because it is already implemented and comparable across existing runs. A future stakeholder-facing version should support an `inconclusive`/mixed output rather than forcing every branch into recovery or relapse.

## Frozen evidence before repair

### Modern sample

Inputs:

- `data/processed/phase3_spy_vix_volatility_extended.csv`
- `results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv`

Episode geometry:

- 48 complete episodes
- 17 recovery
- 13 relapse
- 18 mixed
- 17 binary OOS predictions under the current prequential protocol

Direct transition results:

| Model | ROC AUC | Brier |
|---|---:|---:|
| historical class rate | 0.492 | 0.254 |
| VIX only | 0.386 | 0.268 |
| current state | 0.500 | 0.304 |
| state + motion | 0.621 | 0.275 |
| HAR forecast gap | 0.364 | 0.270 |
| HAR + motion | 0.591 | 0.290 |
| full 40-day path linear | 0.379 | 0.402 |
| reset ESN ensemble | 0.394 | 0.329 |
| continuous ESN ensemble | 0.409 | 0.328 |

Interpretation:

- VIX, static state, and HAR-derived volatility forecasts do not resolve transition direction.
- The only cheap direct signal found is in recent motion.
- The primitive temporal models do not preserve or improve that signal.
- This does **not** establish that temporal information is absent because the ESN architecture was not calibrated for the new target.

### Canonical long history

Inputs:

- 1950-01-03 through 2026-07-02
- 80 complete episodes
- 28 recovery
- 19 relapse
- 33 mixed
- 47 binary episodes total
- 37 binary OOS predictions

Frozen direct replication:

| Model | ROC AUC | Brier |
|---|---:|---:|
| historical class rate | 0.395 | 0.251 |
| state + motion | 0.403 | 0.255 |
| current state | 0.364 | 0.262 |
| continuous ESN ensemble | 0.345 | 0.329 |
| reset ESN ensemble | 0.348 | 0.329 |
| full 40-day path linear | 0.279 | 0.416 |

The label orientation was verified:

- recovery = 1
- relapse = 0

The long-history task is strongly nonstationary. OOS recovery rates by decade vary sharply, including approximately 0.27 in the 2000s versus approximately 0.83 in the 2010s.

Subperiod AUCs show that `state_plus_motion` becomes more useful in the contemporary period:

| Model | pre-2000 | 2000+ | 2010+ |
|---|---:|---:|---:|
| state + motion | 0.033 | 0.438 | 0.583 |
| full path linear | 0.333 | 0.208 | 0.056 |
| reset ESN ensemble | 0.500 | 0.299 | 0.389 |
| continuous ESN ensemble | 0.467 | 0.312 | 0.417 |

Combined with the modern-sample `state_plus_motion` AUC of 0.621, this suggests that recent motion is more relevant to contemporary branch resolution, while the current temporal representations may encode unstable historical/era structure.

## Architecture audit

The current branch ESN is a smoke test, not a serious benchmark:

- 50 reservoir units
- one spectral radius: 0.9
- one input scale: 0.3
- one leak: 0.3
- three fixed seeds
- no architecture calibration
- fixed logistic readout with `C=0.1`

The historical Phase 2 NumPy ESN used much larger reservoirs and multiple frozen configurations, including 300-500 units and substantially different regularization. Therefore the current result does not test the historically successful ESN architecture on the new target.

### 1. Unequal channel scales before the nonlinear reservoir

The six path channels are clipped but not normalized before entering the reservoir:

- `spy_log_return`: `[-0.15, 0.15]`
- `spy_abs_log_return`: `[0.00, 0.15]`
- `log_rv5_over_rv20`: `[-2.0, 2.0]`
- `log_rv20_over_rv60`: `[-1.5, 1.5]`
- `drawdown_120d_path`: `[-0.60, 0.05]`
- `rv5_change_scaled`: `[-3.0, 3.0]`

All channels then share one global `input_scale=0.3`.

This means large-amplitude channels can dominate the reservoir drive. Standardizing the final reservoir states at the readout cannot recover information that was poorly encoded inside the nonlinear dynamics.

Classification: **serious architectural defect for interpreting the current negative result**.

### 2. No reservoir bias or constant input channel

The update is:

```text
h_t = (1 - leak) h_{t-1} + leak * tanh(W_in u_t + W h_{t-1})
```

There is no bias term and no constant input channel.

Several branch-path inputs are intrinsically asymmetric, including absolute return and drawdown. Without a bias, the nonlinear operating point is constrained around zero.

Classification: **likely architectural weakness**.

### 3. Final-state-only compression discards almost the entire trajectory

For each 40-day reset window, the current ESN keeps only:

```text
[final reservoir state, final raw input]
```

All intermediate reservoir states are discarded.

This was also a previously remembered Claude criticism. It is particularly relevant because the target may depend on path morphology rather than only the branch-point endpoint.

Classification: **likely architectural weakness and direct target of the one-day audit**.

### 4. One homogeneous memory timescale

All units share one leak and one recurrent spectral radius. The input contains daily shocks, 5-day motion, 20-day stabilization, and 40-day path morphology.

A single homogeneous timescale may be inadequate, but this is not the first repair because the one-day project must remain bounded.

Classification: **plausible weakness; defer unless simpler repairs establish a need**.

### 5. Continuous-state ESN is not a clean upper bound on memory

The continuous model runs from the beginning of the daily history and samples the hidden state at each branch point. It therefore carries potentially years of history rather than a controlled branch-precursor memory.

The reset model may forget too much; the continuous model may retain too much irrelevant history. Poor performance of both does not bracket the correct memory design.

Classification: **interpretation limitation**.

## One-day repair plan

The goal is not to optimize the binary target aggressively. The future stakeholder-facing task may become recovery/relapse/inconclusive, so model search is intentionally limited.

Compare only:

1. **Primitive frozen ESN**
   - exact existing implementation

2. **Normalized-input ESN**
   - repair channel-scale imbalance before the reservoir

3. **Normalized-input + bias ESN**
   - repair nonlinear operating point

4. **Normalized-input + bias + trajectory summary ESN**
   - preserve final state plus mean reservoir state across the 40-day window

Initial trajectory feature:

```text
[final state, mean state, final input]
```

Do not add max pooling, multiple reservoir sizes, broad spectral-radius searches, leak searches, or multi-timescale reservoirs unless the bounded variants expose a clear reason.

## Evaluation requirements

For every variant report:

- full 1950-2026 prequential metrics
- 2000+ metrics
- 2010+ metrics
- modern SPY metrics
- individual fixed seeds
- fixed-seed ensemble
- ROC AUC
- Brier score
- log loss

The success criterion is not maximum full-history AUC. The relevant question is:

> Does a minimal architectural repair preserve or improve contemporary transition information consistently enough that the primitive ESN failure should not be trusted?

## Hard stop

Stop after the bounded variants unless one of them reveals a specific, evidence-based memory problem. Do not turn this into a broad ESN optimization campaign.

After a usable version is settled, hand the implementation and results to Claude for independent critique and improvement suggestions.
