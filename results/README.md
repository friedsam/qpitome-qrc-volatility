# Results directories

Every run writes under exactly one dataset root:

```text
results/
├── paper_monthly/
├── volare/
└── legacy_daily/
```

Within each dataset, use the same model stages:

```text
<dataset>/
├── sanity_baseline/
├── esn/
├── tfim/
├── rydberg/
└── comparisons/
```

Rules:

- No shared `current/` directory across datasets.
- Every run records dataset ID, data hash, code commit, split definition, target, feature set, seed(s), selection metric, and evaluation metric.
- Prediction-level files remain local by default; compact summaries are the review artifacts.
- No smoke or routing outputs are mixed with scientific results.
