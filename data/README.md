# Data directories

All data are separated by dataset and stage.

```text
data/
├── raw/
│   ├── paper_monthly/
│   ├── volare/
│   └── legacy_daily/
├── interim/
│   ├── paper_monthly/
│   ├── volare/
│   └── legacy_daily/
└── processed/
    ├── paper_monthly/
    ├── volare/
    └── legacy_daily/
```

Rules:

- `raw/` contains immutable timestamped snapshots plus provenance manifests.
- `interim/` contains transparent derived tables that are not yet approved for modeling.
- `processed/` contains only audited model-ready tables.
- Raw/derived data are ignored by Git by default; curated release artifacts require an explicit decision.
- No script may read one dataset and write into another dataset's directory.
