# Aquila local-detuning runbook (qBraid path)

Last updated: 2026-07-08

This runbook records the complete path needed to construct, validate, submit, diagnose, and retrieve qBraid Aquila programs using Braket AHS local detuning. It exists because the working path is not obvious and several failures occur before hardware execution.

## Scope

This is specifically for:

- qBraid-managed Aquila access;
- device ID `aws:quera:qpu:aquila`;
- Braket AHS `AnalogHamiltonianSimulation`;
- `LocalDetuning` / local shifting field;
- qBraid SDK 0.12.1 behavior observed in the project environment.

Do not substitute direct AWS Braket unless the account actually has AWS credentials and billing configured. The successful project hardware path uses qBraid credits, not direct AWS credentials.

## Known-good environment setup

From the qBraid instance:

```bash
cd ~/qpitome-qrc-volatility
git pull --ff-only
source .venv/bin/activate
which python
```

Expected Python path:

```text
/home/jovyan/qpitome-qrc-volatility/.venv/bin/python
```

## Correct device access path

Use:

```python
from qbraid.runtime import QbraidProvider

device = QbraidProvider().get_device("aws:quera:qpu:aquila")
```

Do not use `AwsDevice(...)` for this project path. Direct AWS construction failed because the qBraid environment had no AWS region and then no AWS credentials. The project credits live on qBraid.

## What the qBraid device profile does and does not tell us

The device profile exposed:

```text
status=DeviceStatus.ONLINE
experiment_type=ExperimentType.ANALOG
program_spec=ProgramSpec(AnalogHamiltonianSimulation, braket_ahs)
```

The profile contains no detailed Hamiltonian capability schema. Therefore, absence of strings such as `localDetuning` in the profile does not prove lack of support.

## SDK support for local detuning

The installed Braket SDK includes:

```text
braket.ahs.local_detuning.LocalDetuning
```

Constructor API:

```python
LocalDetuning.from_lists(times, values, pattern)
```

The local-detuning term can be combined with the ordinary global driving field:

```python
program = AnalogHamiltonianSimulation(
    register=register,
    hamiltonian=drive + local,
)
```

`program.to_ir()` successfully serializes to Braket AHS IR with an explicit `localDetuning` field.

## qBraid pre-submission methods

The qBraid Aquila device exposes:

```text
prepare(run_input)
transform(run_input)
transpile(run_input, run_input_spec)
validate(run_input_batch)
```

The narrowest compatibility check is:

```python
device.validate([program])
```

This accepted a program containing local detuning.

Important: `validate()` is not sufficient to prove the full submission path works. `device.run()` performs additional preparation/serialization and then live backend Hamiltonian validation.

## qBraid 0.12.1 Decimal serialization bug

Observed failure before submission:

```text
TypeError: Object of type Decimal is not JSON serializable
```

Path:

```text
device.run()
→ apply_runtime_profile()
→ prepare()
→ target_spec.serialize()
→ qbraid_program.serialize()
→ AnalogHamiltonianEncoder
```

The qBraid encoder handles `AnalogHamiltonianProgram` itself but not nested `Decimal` values introduced by the local-detuning IR.

Working runtime patch:

```python
from decimal import Decimal
from qbraid.programs.analog._model import AnalogHamiltonianEncoder

_original_default = AnalogHamiltonianEncoder.default


def patched_default(self, obj):
    if isinstance(obj, Decimal):
        return float(obj)
    return _original_default(self, obj)


AnalogHamiltonianEncoder.default = patched_default
```

After this patch:

```python
prepared = device.prepare(program)
```

succeeds and returns a qBraid runtime `Program` with `format='analog'`.

The serialized payload contains the expected `localDetuning` block.

## Experimental capability flag is mandatory

A submitted local-detuning job without experimental runtime options failed immediately with:

```text
Hamiltonian Validation error: Specifying local detuning is an experimental capability.
To enable, pass runtime_options={"experimental_capabilities": "ALL"}
in the device.run() call.
```

Required submission form:

```python
job = device.run(
    program,
    shots=shots,
    runtime_options={"experimental_capabilities": "ALL"},
    tags={...},
)
```

The rejected job had:

```text
executionDuration=0
cost=0.0
```

So it never reached hardware.

## Local-detuning hardware constraints discovered from live validation

The live backend validation revealed constraints not enforced by the earlier generic qBraid validation.

### Magnitude range

Positive local-detuning magnitude is invalid.

Rejected example:

```text
+1.0e6 rad/s
```

Backend error:

```text
Value 1 (1000000.0) in magnitude time_series outside the allowed range
[-125000000.0, 0.0]
```

Therefore, local-detuning magnitude must satisfy:

```text
-125e6 <= magnitude <= 0
```

### Endpoint constraints

The first and last values of the local-detuning magnitude time series must both be zero.

Rejected example:

```text
0 → -1e6
```

Backend error:

```text
The values of the shifting field magnitude time series at the first and last time points
are 0.0, -1000000.0; they both must be both 0.
```

Use a pulse such as:

```text
0 → negative interior value → 0
```

For the smoke test:

```python
local = LocalDetuning.from_lists(
    times=[0.0, 1.5e-6, 3.0e-6],
    values=[0.0, -1.0e6, 0.0],
    pattern=[0.0, 0.33, 0.67, 1.0],
)
```

## Known smoke-test program structure

Global drive:

```text
4 atoms
7 μm spacing
3 μs total duration
Ω max = 6.3e6 rad/s
global detuning = -25.2e6 → +25.2e6 rad/s
```

Local detuning:

```text
magnitude pulse = 0 → -1.0e6 → 0 rad/s
pattern = [0.0, 0.33, 0.67, 1.0]
```

## Current guarded script

Use:

```text
scripts/hardware/run_aquila_local_detuning_smoke_test.py
```

The script should:

1. activate the qBraid Decimal encoder workaround;
2. build the Braket AHS local-detuning program;
3. call `device.validate([program])`;
4. call `device.prepare(program)`;
5. in hardware mode, call `device.run(...)` with `runtime_options={"experimental_capabilities": "ALL"}`;
6. checkpoint the job ID immediately;
7. retrieve the result;
8. write failure metadata instead of assuming measurement counts exist.

Guarded submission command:

```bash
python scripts/hardware/run_aquila_local_detuning_smoke_test.py \
  --hardware \
  --shots 1 \
  --confirm SUBMIT_AQUILA_LOCAL_DETUNING_TEST
```

## Job retrieval

Never resubmit just because the foreground process crashes after job creation.

Retrieve an existing job by ID:

```python
from qbraid.runtime.native.job import QbraidJob

job = QbraidJob("<JOB_ID>")
print(job.status())
print(job.metadata())
result = job.result()
print(result)
print(result.success)
print(result.data.get_counts())
```

The existing smoke script also supports retrieval by `--job-id`.

## Failure diagnosis

The most useful rejection reason was stored in job metadata rather than `result.data.extra`.

Inspect:

```python
metadata = job.metadata()
print(metadata["status"])
```

Also useful:

```python
print(job.__dict__.get("_cache_metadata"))
```

Do not assume `result.data.get_counts()` exists when a job fails. Failed jobs returned:

```text
AnalogResultData(measurement_counts=None, measurements=None)
```

## Cost behavior observed

Pre-hardware validation failures observed in this sequence cost:

```text
0.0 qBraid credits
```

The qBraid metadata estimated a one-shot local-detuning task at:

```text
31.0 credits
```

This may change; treat the current runtime metadata as authoritative.

## Jobs from the discovery sequence

### Experimental capability missing

```text
aws:quera:qpu:aquila-6e03-qjob-6a4da40c4620e1f69885aea2
```

Outcome:

```text
FAILED
cost=0.0
reason=experimental_capabilities flag missing
```

### Positive local-detuning magnitude

```text
aws:quera:qpu:aquila-6e03-qjob-6a4da53f4620e1f69885aeb6
```

Outcome:

```text
FAILED
cost=0.0
reason=local magnitude +1e6 outside [-125e6, 0]
```

### Nonzero final local-detuning magnitude

```text
aws:quera:qpu:aquila-6e03-qjob-6a4da5da4620e1f69885aec4
```

Outcome:

```text
FAILED
cost=0.0
reason=first and last local magnitude values must both be zero
```

## Current final gate

The corrected program uses:

```text
runtime_options={"experimental_capabilities": "ALL"}
local magnitude pulse 0 → -1e6 → 0
qBraid Decimal serialization workaround
```

At the time this runbook was written, the corrected job had reached the Aquila queue. A queued job means it passed:

```text
Braket LocalDetuning construction
→ Braket AHS IR serialization
→ qBraid device validation
→ qBraid prepare/serialization workaround
→ experimental capability authorization
→ live Hamiltonian validation
→ Aquila queue
```

Record the final successful job ID and result here once available.

## Minimal checklist for next time

```text
[ ] Use qBraid provider, not direct AwsDevice
[ ] Activate project .venv
[ ] Patch Decimal serialization for qBraid 0.12.1 if still needed
[ ] Local magnitude is nonpositive
[ ] Local magnitude starts at 0
[ ] Local magnitude ends at 0
[ ] Pass runtime_options={"experimental_capabilities": "ALL"}
[ ] Call validate()
[ ] Call prepare()
[ ] Submit only after both pass
[ ] Checkpoint job ID immediately
[ ] On failure, inspect job.metadata()["status"] before resubmitting
[ ] Retrieve existing jobs instead of submitting replacements
```
