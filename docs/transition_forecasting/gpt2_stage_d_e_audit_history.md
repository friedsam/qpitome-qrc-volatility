# GPT-2 Stage D/E Audit History

This document preserves the staged repo audit that established the final Stage D/Stage E chronology design. It records both the findings and the reasoning sequence so later refactors do not remove safeguards whose necessity is not obvious from the final code alone.

## Audit method

The audit proceeded in batches:

1. Identify the exact Stage E interface and invariants.
2. Verify split, metric, scaling, and reporting discipline.
3. Separate confirmed guarantees from unresolved leakage questions.
4. Recover and inspect the Stage D builder and saved results.
5. Quantify chronology defects directly from sample windows.
6. Repair fold construction without discarding canonical Stage D.

The governing rule was: do not convert an unresolved concern into a claim before locating the producing code or artifact.

## Batch 1: interface and first chronology concern

Verified Stage E inputs:

- `sample_manifest.csv`
- `sequence_tensors.npz`
- tensor shape `(n_samples, 40, 1)`
- exact manifest/tensor `sample_id` order alignment
- ten continuous targets `target_x_h1 ... target_x_h10`

Verified split and evaluation discipline:

- episodes cannot span train/validation/test;
- missing splits fail;
- development screens use validation only;
- outputs explicitly record `"test_evaluated": false`.

Verified QLIKE implementation for log-volatility predictions:

\[
\ell = \frac{\sigma_t^2}{\hat{\sigma}_t^2}
- \log\left(\frac{\sigma_t^2}{\hat{\sigma}_t^2}\right) - 1
\]

with numerical clipping.

Reporting was stratified by pooled samples, label, lead, lead × label, each horizon, and full path.

Initial baselines:

- persistence;
- last-level ridge;
- HAR ridge;
- extended HAR ridge;
- full-sequence ridge;
- independently shuffled-sequence ridge.

All ridge models used training-fitted standardization. Alpha was selected on validation QLIKE.

The initial rolling design sorted episode groups chronologically, reserved the newest 17% as fixed test, used expanding-train/next-block-validation folds, and applied a 10-calendar-day embargo. Test was not scored.

The first unresolved concern was that episode exclusivity plus a 10-day embargo did not prove disjoint underlying observations for a sample spanning 40 input sessions and 10 target sessions. No larger embargo was assumed before the Stage D builder was inspected.

## Batch 2: split and leakage geometry

Verified saved Stage D contract:

- 4,572 samples;
- 198 episodes;
- 40-step inputs;
- 10-step targets;
- 1,143 positives;
- 3,429 controls;
- whole-episode static split assignment;
- untouched test in recent screens.

Existing checks proved:

1. manifest/tensor alignment;
2. no episode split overlap;
3. train-only scaling;
4. chronological episode-level folds;
5. 10-day boundary embargo;
6. fixed newest 17% test block.

The key distinction was:

| Question | Status |
|---|---|
| Can the same episode cross partitions? | No — verified. |
| Can neighboring samples reuse underlying market dates? | Not yet verified. |

The correct audit unit was the actual interval:

- `input_start_date`
- `origin_date`
- `target_end_date`
- `index`
- `episode_id`
- fold assignment

Required overlap checks included input-input, train-input/validation-target, train-target/validation-input, target-target, and same-global-event representation across markets.

The 10-day embargo was not declared wrong. It remained an unresolved design question pending direct inspection of Stage D and saved outputs.

## Batch 3: direct results audit and confirmed chronology defect

Once the saved results archive was available, a real chronology defect was confirmed.

Rolling-fold assignment used each sample's `episode_id` and positive `event_onset`. Controls inherited the matched positive's episode and split even when the control's own `origin_date` belonged to a much later period.

Examples:

```text
event_onset: 2000-09-06
control origin_date: 2010-11-30
fold assignment: validation for the 1998–2001 block
```

```text
event_onset: 2008-01-10
control origin_date: 2010-06-08
fold assignment: validation for the 2001–2008 block
```

Quantified future-period controls entering earlier training folds:

| Fold | Controls | Fraction of training set |
|---|---:|---:|
| 1 | 229 | 34.7% |
| 2 | 638 | 37.4% |
| 3 | 463 | 17.3% |

Exact duplicated `(index, origin_date)` keys across train and validation:

| Fold | Duplicates |
|---|---:|
| 1 | 3 |
| 2 | 6 |
| 3 | 7 |

A broader 70-calendar-day same-index proximity screen found 17.3%, 40.7%, and 28.6% for folds 1–3. This was treated only as a screening statistic; the chronology inversion itself was already proven.

The original static split was safer, but control origins were still not guaranteed to lie inside the inherited static period. Static splits therefore also required origin-date and interval auditing.

The saved rolling and nested rolling results were not valid strict chronological-generalization estimates. This did not prove the signal fictitious; it required re-evaluation of ordered-vs-shuffled robustness, nested alpha effects, fold-specific ESN behavior, cross-market comparisons, lead-5 concentration, and the initial validation ladder.

## Root cause

Control matching was correct cross-sectionally:

- same index;
- same lead;
- matched positive episode and split;
- three controls per positive.

But matching did not constrain the control's actual market window to the chronology of the fold. Balance was preserved while rolling chronology was broken.

## Correct repair boundary

Do not discard canonical Stage D. Repair only fold construction:

1. Assign positives chronologically by episode.
2. Define each fold's allowed calendar interval.
3. Require each control's complete input and target interval to lie inside that interval.
4. Rematch controls within the permitted fold-local candidate pool.
5. Deduplicate `(index, origin_date)` where required.
6. Purge overlapping same-index intervals across train and validation.
7. Re-run compact classical survivors using fixed-alpha or properly nested selection.
8. Keep test untouched.

The correct splitting unit is:

```text
input_start_date -> origin_date -> target_end_date
```

not merely the positive episode label.

## Resolution map

| Finding | Resolution |
|---|---|
| Episode IDs do not prove raw-window separation | Persist exact interval columns. |
| Controls can come from future periods | Fold-local control rematching. |
| Duplicate control origins can cross partitions | Deduplication and origin-reuse safeguards. |
| A fixed embargo cannot prove interval disjointness | Purge by actual same-index interval intersection. |
| Per-fold best-alpha aggregation is optimistic | Fixed-alpha or properly nested selection. |
| Test contamination risk | Preserve `test_evaluated: false` and fixed untouched test. |

Later relevant refinements included exact trading-row intervals, strict sample chronology, partition-local control rematching, full candidate pools, fold-local reconstructed datasets, and prevention of inappropriate control-origin reuse across leads.

These are downstream evaluation protections. They are not replacements for canonical Stage D construction.

## Permanent invariants

- Canonical tensors remain `(n_samples, 40, 1)` with exact manifest alignment.
- Targets remain ten continuous future log-volatility values.
- Episode exclusivity is necessary but insufficient for rolling chronology.
- Rolling and nested evaluation must use actual sample intervals.
- Controls must be rematched within each fold's permitted calendar range.
- Same-index train/validation overlap must be purged explicitly.
- Test remains untouched during development.
- Historical pre-repair rolling results must not be described as strict chronological-generalization estimates.

## Why this history is retained

The final safeguards can look redundant when viewed only in the current code. They arose from a measured defect in saved results. This audit history prevents future refactors from reverting to episode-label-only chronology and silently reintroducing future-period controls into earlier folds.
