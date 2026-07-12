# Project state and submission lineage

Last updated: 2026-07-12

This is the authoritative project-level record. Detailed experiment history remains in topic documentation and Git history.

## Current scientific thesis

The primary stakeholder question is:

> Can we detect, early enough to matter, that the market is entering a rare transition regime where ordinary volatility dynamics become unreliable, and then predict whether the market moves toward recovery or further deterioration?

The project now separates two tasks:

1. **Task A — branching-onset detection:** predict the onset of a rare broad-regime transition.
2. **Task B — destination prediction:** predict the destination regime once the market is entering a branching interval.

A useful Rydberg result may improve either task. Task B is currently the preferred primary target because nonlinear temporal memory is more likely to matter for destination prediction. Task A is retained as a bounded fallback and write-up component.

No quantum advantage has been demonstrated.

## Current route

```text
continuous weekly returns
    -> causal HMM/GARCH classical baseline
    -> identify rare broad-regime transition windows
    -> audit whether onset detection is trivial
    -> define a destination target from future observed market behavior
    -> compare strong classical temporal controls
    -> test temporal Rydberg reservoir on the unresolved residual task
    -> only then perform noise, shot, and hardware studies
```

The previous barrier-defined recovery/relapse classifier is no longer the primary route. It remains part of the historical experiment record but should not be used as the headline target.

## Classical baseline status

### Weekly Gaussian regime models

Causal expanding-history evaluation on weekly S&P 500 returns used 520 initial training weeks and annual refits. The current implementation is a masked Baum-Welch EM approximation, not a faithful Bayesian replication of Maheu, McCurdy, and Song.

One-step predictive-density results over 3,471 forecast weeks:

| Model | Mean log predictive density | Total log predictive density |
|---|---:|---:|
| restricted HMM4 | -2.07638 | -7207.12 |
| unrestricted HMM4 | -2.07859 | -7214.77 |
| HMM2 | -2.09757 | -7280.66 |
| iid Gaussian | -2.21780 | -7697.98 |

Interpretation:

- Four states materially outperform two states and an iid Gaussian baseline.
- The restricted HMM4 beats unrestricted HMM4 by only 7.65 total log-score units.
- That restricted-model advantage is concentrated in the most extreme 1% of return weeks and is strongly influenced by COVID.
- Unrestricted HMM4 remains the neutral predictive comparator.
- Restricted HMM4 remains valuable because its states are economically interpretable and it is more robust in a small number of extreme periods.

### Restricted HMM4 state interpretation

Aggregate causal filtered-state profiles align with the intended four-state hierarchy:

| State | Interpretation | Weighted mean return | Weighted mean absolute return | Weighted 13-week volatility |
|---|---|---:|---:|---:|
| 0 | bear/stress | -0.13% | 2.43% | 2.56% |
| 1 | bear rally | +0.41% | 1.48% | 2.00% |
| 2 | bull correction | -0.46% | 1.65% | 1.74% |
| 3 | bull | +0.48% | 1.10% | 1.67% |

The unrestricted model separates negative states reasonably but produces two less clearly differentiated positive states.

Annual parameter stability is unresolved because the first long run lost fit-history output at manifest serialization. The runner is fixed and now checkpoints `fit_history.csv`. A later overnight rerun is justified, but it is not the immediate priority.

### GARCH

GARCH remains required as a challenge-aligned classical comparator. The baseline set should remain compact:

- iid or persistence baseline;
- GARCH(1,1), preferably Student-t innovations;
- HMM2;
- unrestricted HMM4;
- restricted HMM4;
- one matched classical temporal reservoir for the final Rydberg comparison.

Do not expand classical exploration unless it changes target choice, hardware choice, encoding choice, or the final comparison.

## Task A — branching-onset detection

Exploratory direction-neutral label:

- current broad regime is bear-side `{0,1}` or bull-side `{2,3}`;
- positive label means the broad regime flips within four weeks;
- the new regime persists for at least three of the following four weeks.

The label is retrospective, but all detector features are causal. The final write-up must state that this is forecasting a future latent-regime transition, not an externally observed event label.

Chronological 60/40 holdout audit:

- train prevalence: 0.1413;
- test prevalence: 0.1497;
- strongest trivial detector, HMM long-regime uncertainty: AP 0.3357, ROC AUC 0.7483;
- one-week probability motion: AP 0.2870, ROC AUC 0.7312;
- 4-week volatility: AP 0.1742;
- 13-week volatility: AP 0.1728;
- volatility change: AP 0.1588;
- absolute return: AP 0.1743.

Conclusion:

> Onset prediction is not merely a volatility-threshold problem. HMM state uncertainty contains useful but incomplete warning information.

Task A passes the cheap gate. It should remain a limited write-up component and fallback target, not consume the main project unless Task B fails quickly or extra time remains.

The next Task A check is an event-time trajectory: warning probability should rise as the transition approaches, and useful lead time matters more than same-week confirmation.

## Task B — destination prediction

This is the preferred Rydberg target.

The desired question is:

> Given a causal transition context, can the model predict whether the market moves toward recovery or further deterioration?

Requirements before implementation:

- destination outcome must be defined from future observed market behavior, not solely from an HMM-generated label;
- admission into the transition set must be direction-neutral;
- no shared geometric construction between diagnostic and outcome;
- enough positive and negative episodes must remain for chronological evaluation;
- trivial direction rules must be audited before Rydberg work;
- the final Rydberg comparison should use a frozen classical baseline plus a protected correction whenever practical.

## Rydberg hypothesis

The defensible hypothesis is:

> A temporal Rydberg reservoir can encode nonlinear path-dependent transition structure that is poorly represented by GARCH and finite-state Markov models, improving either early transition warning or conditional destination prediction.

The project should not claim that generic quantum dynamics forecast volatility better. Volatility is retained as an input, comparator, and challenge-aligned diagnostic, not as the sole scientific goal.

## Closed or demoted lanes

### Barrier-defined D1 classifier

The four D1 features are rank two after standardization and nearly algebraically redundant. A zero-parameter first-passage null reproduces most of the D1 log-loss improvement over the prior. The old D1 headline is retired as construction-implied predictability.

### Static and residual Rydberg assays on D1

Occupations and pair observables largely reconstruct D1 geometry. Strict residualized Rydberg confirmation did not add generalizable signal. Higher-order output engineering also failed.

### Generic ESN on the old branch task

Generic ESN representations, PCA compression, and residual correction did not improve the old recovery/relapse task. These remain methodological negative results, not evidence against temporal reservoirs on a redesigned dense or transition target.

## Immediate priorities

1. Record Task A event-time score trajectories without expanding feature engineering.
2. Add GARCH to complete the compact classical baseline.
3. Define and audit Task B for sample count, class balance, trivial predictability, and chronological evaluability.
4. Choose one Rydberg target.
5. Freeze the baseline ladder and evaluation protocol.
6. Run Rydberg simulation, matched classical controls, noise studies, and hardware tests.
7. Write the submission while experiments are still running.

## Repository rules

Use one topic taxonomy across code, results, and documentation:

```text
scripts/modeling/<experiment>.py
results/<family>/<experiment>_vN/
docs/experiments/<protocol_or_results>.md
```

Each durable experiment should have:

- one explicit question;
- one output directory;
- a manifest or configuration record;
- predictions where applicable;
- summary metrics;
- a keep/skip decision.

Do not turn the repository into a list of speculative branches. The final submission should contain one problem formulation, one locked baseline ladder, one selected Rydberg target, and concise challenge-relevant comparisons.
