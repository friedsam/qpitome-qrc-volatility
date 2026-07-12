# Weekly regime-transition redesign

Last updated: 2026-07-12

## Purpose

Replace the artifact-dominated barrier classifier with a challenge-aligned transition problem that gives temporal Rydberg dynamics a real task.

Stakeholder question:

> Can we detect, early enough to matter, that the market is entering a rare transition regime where ordinary volatility dynamics become unreliable, and then predict whether the market moves toward recovery or further deterioration?

## Task split

### Task A — branching-onset detection

Binary early-warning task:

```text
ordinary regime -> rare transition onset
```

Current exploratory label:

- broad bear regime: states `{0,1}`;
- broad bull regime: states `{2,3}`;
- positive label if the broad regime flips within four weeks;
- destination regime must persist for at least three of the next four weeks.

This label is retrospective and direction-neutral. Detector inputs remain causal.

Task A is retained as:

- a write-up component;
- a fallback Rydberg target;
- a possible success criterion if onset can be detected early and nontrivially.

It should not receive extensive feature engineering unless Task B collapses or time remains.

### Task B — destination prediction

Conditional prediction task:

```text
transition context -> recovery versus further deterioration
```

This is the preferred Rydberg target. The final outcome should be derived from future observed market behavior rather than solely from future HMM labels.

## Baseline ladder

| Baseline | Question answered | Failure mode tested | Keep/skip decision |
|---|---|---|---|
| iid Gaussian / persistence | Is any temporal structure needed? | apparent gains from unconditional scale | Keep |
| GARCH(1,1), preferably Student-t | Does conditional variance persistence solve the task? | Rydberg merely relearns volatility clustering | Keep |
| HMM2 | Are coarse bear/bull regimes sufficient? | unnecessary four-state complexity | Keep |
| unrestricted HMM4 | Do four latent components improve prediction without topology assumptions? | restricted model wins only by imposed structure | Keep as neutral HMM comparator |
| restricted HMM4 | Does literature-informed topology improve interpretation or tail robustness? | unrestricted states are unstable or semantically weak | Keep as structural variant |
| matched classical temporal reservoir | Does generic nonlinear memory explain any Rydberg gain? | quantum model wins only against static baselines | Keep for final comparison |
| broad classical hyperparameter search | Can another classical family gain marginally? | project drifts into baseline optimization | Skip unless target choice changes |

## Weekly HMM evidence

Causal expanding-history evaluation:

- weekly S&P 500 returns;
- 520-week initial training history;
- annual refits;
- filtered states only;
- one-step predictive density;
- 3,471 forecast weeks from 1960-01-01 through 2026-07-03.

| Model | Mean log predictive density | Total log predictive density |
|---|---:|---:|
| restricted HMM4 | -2.07638 | -7207.12 |
| unrestricted HMM4 | -2.07859 | -7214.77 |
| HMM2 | -2.09757 | -7280.66 |
| iid Gaussian | -2.21780 | -7697.98 |

Exact differences:

- restricted minus unrestricted HMM4: `+7.6538`;
- restricted minus HMM2: `+73.5465`;
- restricted minus iid Gaussian: `+490.8616`.

Interpretation:

- four states materially outperform two;
- the exact restriction adds only a small average gain;
- restricted-versus-unrestricted advantage is concentrated in the most extreme 1% of return weeks;
- COVID contributes `+7.7105` restricted-versus-unrestricted log-score units;
- unrestricted HMM4 remains the neutral predictive baseline;
- restricted HMM4 remains useful for interpretation and extreme-event robustness.

## Restricted-state interpretation

| State | Intended interpretation | Weighted mean return | Positive-week fraction | Weighted abs. return | Weighted 13-week volatility |
|---|---|---:|---:|---:|---:|
| 0 | bear/stress | -0.132% | 0.489 | 2.427% | 2.558% |
| 1 | bear rally | +0.406% | 0.625 | 1.476% | 2.002% |
| 2 | bull correction | -0.457% | 0.383 | 1.650% | 1.738% |
| 3 | bull | +0.480% | 0.674 | 1.102% | 1.670% |

The restricted model separates sign and volatility in the intended hierarchy. The unrestricted model separates negative states but produces two less cleanly differentiated positive states.

Caveat:

- aggregate profiles do not prove stability across annual refits;
- fit-history recovery requires a checkpointed rerun;
- the implementation is masked Baum-Welch EM, not the paper's Bayesian posterior-predictive method.

## Task A triviality audit

Chronological split:

- total usable weeks: 3,457;
- training: 2,074 weeks through 1999-11-12;
- holdout: 1,383 weeks from 1999-11-19 through 2026-05-15;
- holdout onset prevalence: `0.14967`.

| Detector | Holdout AP | ROC AUC | Precision | Recall | Decision |
|---|---:|---:|---:|---:|---|
| HMM long-regime uncertainty | 0.3357 | 0.7483 | 0.320 | 0.430 | strong natural baseline, task not solved |
| one-week HMM probability motion | 0.2870 | 0.7312 | 0.333 | 0.502 | keep |
| HMM state entropy | 0.2076 | 0.6136 | 0.186 | 0.720 | weak/high false alarms |
| absolute return | 0.1743 | 0.5298 | 0.149 | 0.976 | nearly trivial always-positive rule |
| 4-week realized volatility | 0.1742 | 0.5311 | 0.159 | 0.372 | weak |
| 13-week realized volatility | 0.1728 | 0.5599 | 0.177 | 0.792 | weak/high false alarms |
| volatility change | 0.1588 | 0.4895 | 0.148 | 0.440 | no useful signal |

Conclusion:

> Branching onset is not equivalent to a contemporaneous volatility threshold. HMM uncertainty contains meaningful but incomplete warning information.

Task A passes the cheap gate.

## Task A trajectory check

The next and likely final near-term Task A analysis is event-time profiling.

Required plot/table:

- align confirmed transitions at event time `0`;
- show mean and quantiles of the warning score at leads `-8` through `0` weeks;
- compare transition episodes with matched non-transition weeks;
- report the first lead at which the score exceeds the frozen warning threshold;
- report episode-level detection rate and repeated false alarms.

Keep Task A only if confidence rises before the transition becomes obvious. Same-week detection alone is not a useful early-warning result.

## Task B design requirements

The destination target must satisfy all of the following:

1. admission into the transition set is direction-neutral;
2. destination is determined from future observed returns or another externally interpretable market quantity;
3. admission and outcome do not share a construction-implied geometry;
4. evaluation is chronological and grouped by transition episode;
5. class counts are sufficient for at least a bounded comparison;
6. simple return, volatility, HMM, and GARCH rules are audited first;
7. Rydberg is tested only after one target passes this gate.

## Rydberg decision rule

Promote one task to the full Rydberg program only when:

- the label is defensible;
- the task is not solved by a trivial diagnostic;
- the sample size supports a bounded claim;
- a strong classical baseline is frozen;
- there is a specific nonlinear temporal residual for the reservoir to address.

A valid Phase 3 success can be:

- improved early branching-state detection; or
- improved conditional destination prediction.

The model does not need to win both tasks.

## Files

Current scripts:

```text
scripts/modeling/run_weekly_regime_baselines.py
scripts/modeling/analyze_weekly_regime_baselines.py
scripts/modeling/analyze_weekly_regime_states.py
scripts/modeling/audit_branch_onset_triviality.py
```

Current HMM implementation:

```text
src/qpitome_qrc/baselines/gaussian_hmm.py
```

Temporary outputs from the first complete run:

```text
/tmp/weekly_regime_baselines/
/tmp/weekly_regime_baseline_diagnostics/
/tmp/weekly_regime_state_diagnostics/
/tmp/branch_onset_triviality_audit/
```

Durable reruns should move to a versioned `results/regimes/` directory with manifest, predictions, fit history, and summary metrics.

## Keep/stop decisions

- **KEEP:** compact HMM baseline ladder.
- **KEEP:** GARCH as the next classical comparator.
- **KEEP, bounded:** Task A and its event-time trajectory.
- **PRIORITIZE:** Task B target construction and triviality audit.
- **STOP:** further Task A feature engineering for now.
- **STOP:** barrier-defined classifier as the primary route.
- **STOP:** broad classical model proliferation.
- **DEFER:** overnight HMM parameter-stability rerun until it does not interrupt target work.
