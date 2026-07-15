# Data layout

- `raw/`: datasets fetched directly from upstream providers.
- `processed/`: deterministic Stage 1 datasets built from `raw/`.
- `fallback/`: frozen copies of successfully fetched raw datasets.
- `manifest.json`: authoritative inventory and provenance record.

Data-fetching and processing logic lives under `scripts/data/` and `src/data/`.
