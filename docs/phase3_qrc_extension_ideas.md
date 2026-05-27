# Phase 3 QRC Extension Ideas

This note preserves promising follow-up directions from the Phase 2 QRC volatility experiments. The current Phase 2 result should be frozen around the best leaky-input TFIM-QRC unless a very small bounded test produces a clear improvement.

## Current best Phase 2 QRC reference

Architecture:

```text
6-qubit full-topology TFIM-QRC
PCA-6 volatility features
40-day lookback
10 recent anchors
leaky-integrated PCA input, leak = 0.3
3 virtual nodes per anchor
3 Trotter steps per anchor
evolution_time = 0.5
ZXZZ observables
disorder_strength = 0.20
winsorized top-120 ridge readout
log target
```

Observed test performance from the leaky-input probe:

```text
test_rmse          ≈ 0.096106
test_qlike         ≈ -2.224603
test_mz_r2         ≈ 0.166418
test_corr          ≈ 0.407943
high-vol recall    ≈ 0.549180
```

Interpretation:

```text
Leaky trajectory encoding improved the exact failure mode exposed by the ESN comparison: compressed predictions and weak high-volatility recall.
```

## Lessons from Phase 2

1. The reset-window ESN beats reset-window QRC, so cross-window carryover is not the primary explanation of the gap.
2. ESN acts as a useful teacher: its advantage comes from stronger within-window trajectory processing and more target-aligned reservoir states.
3. Sparse raw angle-encoded QRC anchors lose trajectory-shape information.
4. Recent-biased anchors help more than simply increasing anchor count.
5. Leaky-integrated inputs substantially improve QRC high-volatility recall and all major test metrics.
6. Encoding is the main active design axis for future work.

## Small Phase 2 improvements still worth considering

These should be treated as bounded tests only, not open-ended tuning.

### 1. Tanh-angle encoding

Replace clipped linear angle scaling with a saturating nonlinear map:

```text
theta = angle_max * tanh(scale * x)
```

Rationale: ESN benefits from saturating nonlinear tanh dynamics at every timestep. This test keeps qubit count fixed while making the QRC input map less locally linear.

Suggested sweep:

```text
scale ∈ {0.5, 1.0, 1.5}
base input = leaky PCA with leak = 0.3
base QRC = 10 recent anchors, full TFIM, virtual nodes
```

Decision metrics:

```text
test RMSE
test QLIKE
test MZ R²
test prediction std
high-vol recall / precision
top-quintile bias
```

### 2. Very small readout sensitivity around the best leaky-input QRC

Only if time permits:

```text
top_k ∈ {80, 120, 160, 240}
alpha ∈ {1000, 3000, 10000}
```

Rationale: leaky input changed feature geometry; previous top-k/alpha settings may not be optimal.

Guardrail: stop if improvements are tiny or inconsistent across RMSE, QLIKE, MZ R², and high-vol recall.

### 3. Top-quintile residual diagnostics

Not a model change. Add reporting of volatility-quantile bias and RMSE to the final analysis:

```text
actual-volatility quantile bins
mean actual volatility
mean predicted volatility
bias
RMSE
```

Rationale: QRC’s main failure mode has been underprediction of high-volatility regimes. The final report should show whether leaky input reduces top-quintile underprediction.

## Phase 3 extension ideas

### 1. Delta / momentum encoding with dimension control

Encode both level and change while preserving 6-qubit input dimension.

Candidate design:

```text
3 PCA level components + 3 PCA delta components
```

or train-only PCA on concatenated level/delta features:

```text
[level_t, level_t - level_{t-1}] -> PCA-6 -> angle encoding
```

Purpose: explicitly expose volatility acceleration to the QRC.

### 2. Tanh-angle / nonlinear encoding family

Systematically compare input maps:

```text
linear clipped angle
standardized tanh angle
rank/quantile angle
signed square-root angle
```

Purpose: determine whether QRC needs stronger nonlinear input activation to detect high-volatility regimes.

### 3. Data re-uploading QRC

At each anchor, encode the same input more than once with short TFIM evolution between uploads:

```text
encode x_t -> evolve -> encode x_t -> evolve -> measure
```

Purpose: increase nonlinear input-reservoir mixing without increasing qubit count.

### 4. Streaming / carryover QRC

Run QRC sequentially across chronological samples, resetting only at train/val/test split boundaries.

Purpose: test whether persistent quantum state helps once the reset-window encoding bottleneck has been addressed.

Important caveat: this is an extension, not the immediate explanation for the Phase 2 gap, because reset-window ESN already outperformed reset-window QRC.

### 5. Residual target modeling

Train QRC to predict residual volatility beyond a classical baseline:

```text
future volatility - persistence
future volatility - HAR prediction
future volatility - lightweight ESN prediction
```

Purpose: ask whether QRC contributes a complementary nonlinear signal rather than competing head-to-head as a standalone forecaster.

### 6. Extreme-aware readout

Modify the readout or objective to emphasize high-volatility regimes:

```text
sample weighting by realized volatility
quantile-weighted loss
classification-assisted regression
separate high-volatility calibration layer
```

Purpose: directly address QRC’s historical underprediction of high-volatility episodes.

### 7. Regime-shift classification, with fair ESN baseline

Convert the task to predicting transitions into high-volatility states.

Guardrail: must include strong classical baselines, especially an ESN classifier, to avoid making the task easier only for QRC.

Candidate labels:

```text
future_rv_20d above train 80th percentile
future_rv_20d crossing from below to above train 80th percentile
large positive delta in future realized volatility
```

### 8. Hardware-aware / provider-aware implementation

Map the final small QRC circuit family to a realistic provider stack:

```text
qBraid workflow
Qiskit / Braket compatible circuit export
shot-based readout analysis
noise sensitivity
resource estimates
```

Purpose: strengthen the hybrid integration and deployment story.

## Recommended final Phase 2 stopping point

Freeze Phase 2 around the best leaky-input QRC unless one bounded tanh-angle or readout sensitivity test improves clearly.

Suggested final narrative:

```text
We began with a standard angle-encoded TFIM-QRC baseline. After comparison with a reset-window ESN, we identified encoding as the main weakness: QRC was not capturing within-window trajectory shape. We therefore introduced recent-biased anchors and leaky-integrated trajectory encoding. This improved all major QRC metrics and substantially improved high-volatility recall, although ESN remained stronger overall. The result is a rigorous architecture study rather than an unsupported quantum-advantage claim.
```
