# Kaggle Source Setup for the Transition Data Pipeline

The transition-data workflow retrieves the global stock-index OHLC dataset from Kaggle:

`guillemservera/global-stock-indices-historical-data`

A clean machine must have both the Kaggle command-line client and valid Kaggle authentication before running the live acquisition workflow.

## 1. Install the project dependencies

From the repository root and inside the project environment:

```zsh
python -m pip install -e ".[test]"
```

The project dependencies include the Kaggle client. Confirm that the executable is available:

```zsh
which kaggle
kaggle --version
```

If `which kaggle` prints nothing, confirm that the active Python environment is the same environment in which the package was installed:

```zsh
which python
python -m pip show kaggle
python -m site
```

## 2. Authenticate with Kaggle

Current Kaggle CLI releases support an access-token file at:

```text
~/.kaggle/access_token
```

After generating the token from Kaggle account settings, place it there and restrict access:

```zsh
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
chmod 600 ~/.kaggle/access_token
```

Some Kaggle accounts or older CLI releases instead provide the legacy credentials file:

```text
~/.kaggle/kaggle.json
```

For that method:

```zsh
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

Only one valid authentication method is required. Do not rename a valid `access_token` merely because `kaggle.json` is absent.

Do not commit either credential file to the repository. Both contain private authentication material.

Confirm authentication before running the full pipeline:

```zsh
kaggle datasets files guillemservera/global-stock-indices-historical-data
```

A successful response should list files from the dataset rather than returning an authentication or permission error.

## 3. Run the transition-data workflow

Use a new run ID for each attempt:

```zsh
python scripts/runs/run_submission.py transition-data \
  --run-id transition-data-port-002 \
  --transition-source-mode live
```

The live source is downloaded into the run directory rather than the repository-level `data/raw` tree. The acquisition manifest records the source dataset, retrieval time, file inventory, and hashes.

## 4. Source modes

The runner supports:

- `live`: require a working Kaggle CLI and valid authentication;
- `fallback`: require a local verified fallback snapshot and its `fallback_manifest.json`;
- `auto`: try Kaggle first and use the verified fallback only if the live attempt fails.

The large fallback snapshot is not currently included in `stage1-dev`. On a clean clone, `auto` therefore still requires working Kaggle access unless a fallback snapshot is supplied separately.

## 5. Common failures

### `Kaggle CLI not found`

The Kaggle executable is not available on `PATH` in the active environment. Reinstall the project dependencies in that environment and verify `which kaggle`.

### Neither `~/.kaggle/access_token` nor `~/.kaggle/kaggle.json` exists

The client is installed but no file-based authentication is configured. Generate a Kaggle access token and install the credential file with mode `600`.

### `access_token` exists but `kaggle.json` does not

This is valid for current Kaggle CLI releases. Test it with `kaggle datasets files ...`; no legacy JSON file is required when the access token works.

### Authentication or `401/403` error

The credentials are invalid, expired, incorrectly located, or the dataset terms have not been accepted for the Kaggle account.

### `fallback manifest missing`

The live attempt failed and `auto` then attempted to use a fallback snapshot that is not present. Read the live retrieval error earlier in the acquisition message, configure Kaggle, and rerun with a new run ID.
