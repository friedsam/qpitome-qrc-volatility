# Dataset manifests

Each canonical dataset must have a JSON manifest with, at minimum:

- `dataset_id`
- repository-relative `file`
- `sha256`
- row count and schema version
- date range, when applicable
- producer script
- raw input paths
- source and retrieval date
- originating commit or migration source

A dataset is not canonical merely because a file exists. The manifest and validation checks define the retained version.
