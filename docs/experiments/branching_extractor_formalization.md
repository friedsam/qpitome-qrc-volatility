# Formal branching-state extractor: audit before lock

Last updated: 2026-07-08

## Controlling source

The design is controlled by `docs/experiments/regime_branching_analysis.md`.

The formalization order is:

```text
causal state detection
-> episode construction
-> future outcome labeling
-> episode-level evaluation
-> boring probabilistic baseline
-> residual target
-> only then ESN/QRC tuning
```

The six-class and generic regime-change runs are retained only as diagnostics. They do not define the branch target.

## What the prior work preserved

The exploratory markdown and the independent Claude validation preserve the detector morphology:

```text
high RV20 relative to causal history
persistent drawdown from a trailing peak
short-horizon volatility below longer-horizon volatility or decelerating
recent decline slowed or partially reversed
an actual recent decline must have occurred
```

The validation also preserved:

```text
contiguous candidate days -> episodes
minimum episode separation
40-trading-day outcome horizon
recovery / relapse / mixed labels
24-definition threshold robustness audit
```

The exact exploratory numeric state-threshold grid was not preserved. It must not be fabricated retrospectively.

## Current implementation

Shared module:

```text
src/qpitome_qrc/regimes/branching_state.py
```

Audit runner:

```text
scripts/regimes/audit_branching_extractor.py
```

The module separates three operations.

### 1. Causal state detection

Uses only information available through the candidate date:

```text
RV20 above an expanding historical quantile
120-day drawdown below a threshold
RV5/RV20 below a cap OR 5-day RV change negative
5-day return above a stabilization floor
worst 5-day return in the recent window below a decline threshold
```

No VIX and no future target enters the detector.

### 2. Episode construction

```text
merge nearby positive candidate days
-> enforce minimum episode separation
-> choose one branch point per episode
```

Current branch-point rule:

```text
first qualifying day of the merged causal run
```

Reason: this is the most conservative causal anchor. It does not use later information within the candidate episode to choose the branch point.

This rule is explicit and can be challenged after the audit, but it is not silently optimized against outcomes.

### 3. Future outcome labeling

Future information enters only after episodes exist.

The 40-day label uses the volatility-scaled thresholds supported by the long-history validation:

```text
move_unit = rv20(branch) * sqrt(40 / 252)

recovery threshold = +0.55 * move_unit
relapse threshold  = -0.70 * move_unit
```

Outcome logic:

```text
recovery:
    terminal forward return reaches recovery threshold
    and relapse drawdown threshold is not hit

relapse:
    future running drawdown hits relapse threshold
    and recovery terminal threshold is not reached

mixed:
    both or neither
```

The volatility scaling is important because fixed absolute ±4%/−5% labels were miscalibrated in older lower-volatility eras.

## 24-definition audit

The state threshold grid is:

```text
stress quantile:        0.65, 0.70, 0.75
120d drawdown:         -0.05, -0.06
5d stabilization floor:-0.015, -0.005
recent decline:        -0.03, -0.05
```

This gives:

```text
3 x 2 x 2 x 2 = 24 definitions
```

Fixed across the audit:

```text
RV5/RV20 cap = 1.0
recent-decline lookback = 20 rows
merge gap = 3 rows
minimum episode separation = 20 rows
outcome horizon = 40 rows
```

These numbers are not yet the final benchmark definition. The point of the audit is to determine whether the branching phenomenon is robust to reasonable nearby definitions.

## Audit outputs

Default directory:

```text
results/regimes/branching_extractor_audit_v1/
```

Outputs:

```text
threshold_audit_summary.csv
mid_grid_daily_candidates.csv
mid_grid_episodes.csv
mid_grid_decade_outcomes.csv
run_manifest.json
episode_sets/<one CSV per threshold definition>
```

The audit prints:

```text
all 24 episode/outcome counts
mid-grid episode dates and state values
mid-grid decade outcome counts
```

## Lock criteria

Do not choose the fixed definition solely because it maximizes episode count, class balance, or model performance.

The fixed extractor should satisfy:

1. recurrence across unrelated historical periods;
2. all three outcomes across reasonable nearby definitions;
3. branch dates with the intended unstable-aftermath morphology;
4. no obvious duplicate overlapping episodes;
5. sufficient separation from trivial calm/stress classification;
6. reasonable recovery/relapse/mixed balance under volatility-scaled labels;
7. stability of the scientific conclusion across nearby threshold choices.

Only after the extractor and labels are locked do we construct the episode-level walk-forward benchmark.
