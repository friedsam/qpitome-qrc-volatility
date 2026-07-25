# L5 input/window admission study — 2026-07-25

## Scope

This study uses the frozen plain causal MSE-HAR residual contract. It is restricted to development folds 4–8; folds 4–6 are used for exploration/selection and folds 7–8 for confirmation. Financial test rows are not used.

## Main result

The only representation that produced the requested correction direction on both confirmation folds was the ordered path of the last ten daily changes in log volatility, fitted with a standardized no-intercept Ridge readout.

Frozen exploratory specification:

- input: last 10 changes of the canonical log-volatility sequence;
- readout: multi-output Ridge, alpha 1000, fit_intercept=False;
- target: the ten-horizon frozen causal MSE-HAR residual path.

Confirmation folds 7–8 showed:

- small, bounded mean control corrections near zero;
- positive mean transition corrections through horizon 5;
- transition correction larger than control correction on both folds;
- improved transition QLIKE and RMSE on both folds.

The effect is small and must not be described as a final model or as proof of a quantum advantage.

## Rejected upstream hypotheses

The following did not confirm stably on folds 7–8:

- 40-step absolute level histories;
- level plus local instability;
- direct transition classifiers based on local volatility history;
- local signed price-return features;
- causal global breadth features;
- price-return/volatility-rate phase summaries.

The evidence therefore points to a short, mostly linear temporal signal in recent volatility changes rather than a long calm-history representation or a stable nonlinear cross-channel precursor.

## Task-aligned QRC tests

The rate-window signal was transferred into bounded six-atom reservoir tests using virtual-node measurements at every input step.

The best ordered interacting candidate in the bounded study used:

- one rate channel applied simultaneously to global amplitude and detuning;
- 10 sequential inputs;
- 0.03 microseconds per input;
- interaction scale 1.25;
- all six occupation measurements at every step (60 features);
- no-intercept Ridge, alpha 30000.

It produced small bounded control corrections and positive transition corrections on confirmation folds. It improved transition QLIKE and RMSE on fold 7 and transition RMSE on fold 8, but worsened transition QLIKE on fold 8.

The interaction-off version was numerically almost identical. Additional X/Y readout bases, occupation-pair banks, five-step windows, two-channel price-return/volatility-rate encodings, and the symmetric palindrome did not create a fold-stable interaction-dependent gain. Several shuffled controls ranked above ordered candidates on selection folds.

## Decision

The corrected financial task contains a weak short-memory signal that a classical linear rate-path readout can use. The current reservoir does not yet transform that signal into a fold-stable QLIKE improvement and does not demonstrate a many-body contribution.

A financial hardware run is not justified from this result alone. Any promoted QRC must still satisfy the frozen residual contract, the control/transition correction pattern, ordered-over-shuffled attribution, and interaction-on advantage over the matched interaction-off reservoir.
