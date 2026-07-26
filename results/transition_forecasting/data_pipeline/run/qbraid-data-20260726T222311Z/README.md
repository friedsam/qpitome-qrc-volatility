# Verified transition-data run: `qbraid-data-20260726T222311Z`

- **Experiment:** `global-transition-data-pipeline-fallback-verified`
- **Platform:** `qBraid Lab`
- **Commit:** `19e01623a24634d58b67751453582d8799b66463`
- **Status:** `succeeded`
- **Source mode:** requested `fallback`, used `fallback`
- **Authoritative source verified:** `True`
- **Fallback substitution:** `False`
- **Data audit passed:** `True`
- **Checksum report passed:** `True`
- **Test set evaluated:** `False`
- **Aggregate working run:** `results/runs/qbraid-data-20260726T222311Z`

## Reproduction

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \
  --run-id qbraid-data-20260726T222311Z \
  --transition-source-mode fallback
```

The large aggregate run remains the authoritative execution record. This directory
contains compact manifests, validation reports, command timings, and the required
output inventory needed to inspect and map the run without committing generated
tensors or reconstructed datasets.
