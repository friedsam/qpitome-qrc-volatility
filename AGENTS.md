# Repository Maintenance Rules

These rules apply to every human or automated agent modifying this repository.

## Mandatory preflight

Before changing repository structure, result paths, scripts, documentation, or generated artifacts:

1. Read this file completely.
2. Inspect the relevant directory tree and current Git status.
3. Identify every producer script, consumer script, documentation reference, manifest, test, and workflow affected.
4. Create or update `docs/result_migration_manifest.csv` before moving or renaming any result.
5. State the exact scope of the task. Do not modify files outside that scope.
6. Do not make structural writes until the dependency and provenance map is complete.

## Absolute prohibitions

- Never infer a historical output location from a current script default.
- Never claim an experiment was not run solely because its expected directory is absent.
- Never create a new result hierarchy before inventorying the existing hierarchy.
- Never move, rename, flatten, delete, archive, or overwrite result files without a manifest row for every affected path.
- Never delete unknown, unclassified, or provenance-uncertain files.
- Never silently regenerate missing expensive results and present them as historical originals.
- Never rewrite an entire script when a narrow path edit is sufficient.
- Never change output semantics, model definitions, feature sets, evaluation protocols, or filenames during a path-only maintenance task.
- Never use broad destructive shell commands on `results/`, `data/`, or `scratch/`.
- Never commit generated results unless the repository policy for that result family explicitly requires it.

## Result provenance requirements

Every result family must distinguish:

- historical location actually used by a completed run;
- current default destination for future runs;
- migrated location;
- missing or unverified historical output;
- regenerated output.

For expensive or distributed runs, record:

- script path;
- Git commit SHA;
- full command;
- explicit output directory;
- machine identifier;
- shard index and shard count;
- start and completion time;
- input dataset versions;
- random seed;
- merge command;
- input shard list;
- checksums, file sizes, or row counts.

A later default path is not evidence that an earlier run used that path.

## Migration manifest

`docs/result_migration_manifest.csv` is the authoritative path-migration ledger.

Required columns:

```text
old_path,new_path,producer_script,consumer_scripts,historical_or_new_default,status,provenance_confidence,notes
```

Allowed `status` values:

```text
planned
migrated
preserved
missing
unverified
regenerated
retired
```

Allowed `historical_or_new_default` values:

```text
historical
new_default
both
unknown
```

Allowed `provenance_confidence` values:

```text
high
medium
low
unknown
```

Use semicolons between multiple consumer scripts. Use repository-relative paths. A missing historical artifact must remain recorded even when its current script has a new default destination.

## Path-maintenance workflow

For each result family:

1. Inventory existing files and directories.
2. Read the original producer scripts and their Git history when provenance matters.
3. Search for all path literals and filename references.
4. Map outputs to producers and downstream consumers.
5. Add manifest rows.
6. Make the smallest possible path-only edits.
7. Move files only after the mapping is complete.
8. Verify that expected files exist at the new paths.
9. Search for stale old paths.
10. Compile affected Python files and run relevant tests.
11. Review the diff for accidental semantic changes.
12. Update documentation, including known missing or unverified artifacts.

## Required validation

Run before declaring repository-maintenance work complete:

```bash
python scripts/maintenance/audit_repo_maintenance.py
git diff --check
python -m compileall -q scripts src
```

Also run focused tests for modified code when available.

A maintenance task is not complete if the audit reports errors. Warnings must be reported explicitly rather than omitted.

## Stop conditions

Stop structural work and report the issue when:

- provenance is ambiguous;
- producer and consumer mappings conflict;
- historical files appear missing;
- two different experiments would map to the same destination filename;
- an output directory was supplied dynamically and the historical command is unknown;
- the requested scope would require changing model or evaluation semantics;
- validation reveals unrelated repository damage.

Do not resolve these conditions by guessing.

## Completion report

Every completed maintenance task must report:

- files changed;
- result paths moved or renamed;
- manifest rows added or updated;
- stale-path searches performed;
- validation commands and outcomes;
- missing or unverified artifacts discovered;
- anything intentionally left unchanged.
