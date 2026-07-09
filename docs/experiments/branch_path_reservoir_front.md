# Matched branch path and ESN experiment

Last updated: 2026-07-08

## Question

Does ordered causal pre-branch history add information beyond static state summaries, and does reservoir memory add value beyond direct access to the same path?

## Matched 40-day inputs

All models receive the same causal VIX-blind channels:

```text
SPY log return
absolute log return
log(RV5 / RV20)
log(RV20 / RV60)
120-day drawdown
5-day RV5 change scaled by RV20
```

Each local path contains 40 rows ending at the branch date.

Fixed clipping bounds are applied before any model. They are not fitted on train/test data.

## Models

### Full-path linear control

```text
40 days x 6 channels
-> flatten
-> StandardScaler
-> L2 logistic regression
```

This tests whether simply exposing the full ordered path is enough.

### Reset ESN

Uses the historical NumPy ESN mechanics without changing `numpy_esn.py`:

```text
state reset to zero for each 40-day episode window
-> final reservoir state + final input
-> StandardScaler
-> L2 logistic regression
```

### Continuous-state ESN

The same reservoir update is run once through the entire daily market sequence.

At each branch point:

```text
current reservoir state + current input
-> StandardScaler
-> L2 logistic regression
```

This tests whether memory extending beyond the fixed 40-day local window helps.

## Frozen diagnostic configuration

```text
reservoir units = 50
spectral radius = 0.9
input scale = 0.3
leak = 0.3
seeds = 0, 1, 2
```

The reservoir is intentionally low-dimensional because binary prequential training sets contain only 13–29 episodes. The historical 300–500 dimensional reservoirs would be badly underdetermined for this first classifier diagnostic.

No reservoir tuning is performed against branch outcomes.

For reset and continuous ESN, individual fixed seeds and the unweighted mean probability ensemble are reported.

## Readout

All representations use the same readout:

```text
StandardScaler
-> LogisticRegression(C=0.1)
```

The scaler and classifier are refit on each leakage-safe prequential training set.

## Interpretation

The comparison is hierarchical:

```text
static summaries
vs full-path linear
vs reset ESN
vs continuous-state ESN
```

Possible readings:

- full-path linear wins: ordered path information matters, reservoir unnecessary;
- reset ESN beats full-path linear: nonlinear temporal transformation adds value;
- continuous ESN beats reset ESN: longer-lived state matters;
- no path model improves: branch sample may be too small or the selected channels miss the relevant information.

This is a diagnostic architecture test, not a final tuned ESN benchmark.
