# Aquila hardware case 151

Frozen development example: `P_GE151_^MERV_data_L5`  
Origin: `2014-01-16 00:00:00+00:00`  
Model: six-atom A/B/A palindrome, level + instability, occupation/pair features,
HAR-residual Ridge, `fit_intercept=False`, alpha `0.1`, lambda `0.25`.

The freeze artifact was independently regenerated from the canonical manifest and
tensor archive and matches the archived sample prediction within
`1.750e-13`.

## Run now on qBraid

```bash
python aquila_case151_preflight_submit.py \
  --case case151_freeze.npz \
  --outdir aquila_case151_preflight
```

This does not submit jobs. It queries the live device, attempts Braket
discretization for several stretch factors, and evaluates each hardware-translated
waveform with the frozen exact simulator/readout.

Only after reviewing `aquila_case151_preflight/preflight_summary.csv`:

```bash
python aquila_case151_preflight_submit.py \
  --case case151_freeze.npz \
  --outdir aquila_case151_live \
  --stretch <accepted-factor> \
  --shots 1000 \
  --submit
```

Submission is refused unless all three probe programs discretize and the
hardware-translated exact simulation still improves H5-H10 QLIKE over HAR.

## Files

- `case151_freeze.npz`: encoded input, geometry, exact probabilities/features,
  readout scaler and coefficients, HAR/QRC/realized paths.
- `case151_freeze.json`: provenance and model specification.
- `case151_expected_curve.csv`: curve values for the final before/after figure.
- `case151_encoded_sequence.csv`: raw and scaled 40-lag financial input.
- `case151_geometry.csv`: nominal and interaction-matched hardware coordinates.
- `aquila_case151_preflight_submit.py`: dry-run validation and explicit submitter.

The hardware example is intentionally selected from development data and is not
representative of aggregate forecasting performance.
