# Formal branching-state extractor and outcome-label history

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

Audit runners:

```text
scripts/regimes/audit_branching_extractor.py
scripts/regimes/audit_branch_outcome_labels.py
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

The 40-day label currently uses volatility-scaled thresholds:

```text
move_unit = rv20(branch) * sqrt(40 / 252)

recovery threshold = +0.55 * move_unit
relapse threshold  = -0.70 * move_unit
```

Current provisional outcome logic:

```text
recovery:
    terminal 40-day forward return reaches recovery threshold
    and relapse drawdown threshold is not hit

relapse:
    future running drawdown hits relapse threshold
    and terminal recovery threshold is not reached

mixed:
    both or neither
```

The current 48-episode mid-grid extraction happens to contain only `neither` mixed cases, but `both` remains part of the formal definition.

## Outcome-label history

This section is intentionally retained because the outcome definition may change again.

### Stage A: fixed absolute future-move thresholds

The first exploratory work used fixed absolute recovery/relapse thresholds.

The long-history validation showed that these labels were miscalibrated in older low-volatility eras. A fixed move has different meaning when current realized volatility is 10% versus 40%.

Decision:

```text
reject fixed absolute thresholds as the formal benchmark
```

### Stage B: volatility-scaled thresholds

The long-history validation adopted:

```text
move_unit = rv20(branch) * sqrt(horizon / 252)
recovery = +0.55 * move_unit
relapse  = -0.70 * move_unit
```

This improved outcome balance across eras and is the current basis of the formal labels.

Decision:

```text
retain volatility scaling provisionally
```

### Stage C: terminal recovery versus anytime recovery

The formal implementation initially defined recovery using terminal return at the 40-day horizon.

A focused audit then compared this with an alternative rule:

```text
recovery if the recovery threshold is reached at any time during the 40-day window
```

Current terminal-rule counts:

```text
17 recovery
13 relapse
18 mixed
```

Anytime-recovery counts:

```text
24 recovery
12 relapse
12 mixed
```

Transition table:

```text
mixed -> mixed       11
mixed -> recovery     7
recovery -> recovery 17
relapse -> relapse   12
relapse -> mixed      1
```

All 18 current mixed cases are `neither` cases. None currently hit both thresholds.

The anytime rule was not adopted because it can label a temporary rebound as recovery even when the path later deteriorates. One current relapse became mixed under the anytime rule because both thresholds were reached. That behavior conflicts with the scientific question of genuine recovery versus renewed deterioration.

Decision as of 2026-07-08:

```text
keep terminal-horizon recovery as the provisional main label
keep anytime-recovery results as a documented sensitivity analysis
```

This is not a claim that the terminal rule is permanently correct. Near-threshold cases remain intrinsically brittle. For example, the 2020-03-25 episode finished only slightly below its volatility-scaled terminal recovery threshold.

The benchmark should therefore preserve continuous outcome diagnostics and probability calibration even when hard labels are used for model comparison.

## 24-definition state audit

The state threshold grid is:

```text
stress quantile:         0.65, 0.70, 0.75
120d drawdown:          -0.05, -0.06
5d stabilization floor: -0.015, -0.005
recent decline:         -0.03, -0.05
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

## State-audit result

Across all 24 definitions:

```text
47-78 labeled episodes
all three outcome classes present
episodes recur from the late 1990s through 2025
all definitions span seven distinct five-year blocks
```

The mid-grid candidate currently gives:

```text
48 episodes
17 recovery
13 relapse
18 mixed
```

This is enough to establish robustness of the branching phenomenon and continue the critical path. Whether the mid-grid detector is uniquely preferable is delegated as a parallel robustness question; the main line does not stop on that question.

## Audit outputs

State audit:

```text
results/regimes/branching_extractor_audit_v1/
```

Outcome-label audit:

```text
results/regimes/branch_outcome_label_audit_v1/
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

Current status:

```text
branching phenomenon: robust enough to proceed
mid-grid detector: provisional canonical working definition
outcome labels: provisionally locked for the next benchmark stage
```

The next formal component is the episode-level walk-forward evaluation protocol.
