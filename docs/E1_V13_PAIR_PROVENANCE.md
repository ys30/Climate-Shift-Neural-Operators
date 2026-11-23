# E1 v1.3 — explicit forecast-origin provenance

The ERA5 E1 development pipeline now records both `source_times` and `target_times` for every exported `X`/`y` pair. The data loader checks:

1. Both arrays have exactly one datetime64 value per sample, with no NaT.
2. Source and target timestamps are strictly increasing and unique.
3. For every sample: `target_time - source_time == 24 hours`.
4. The recorded integer year equals **the target year**, even when an origin is in the preceding calendar year.
5. The audit report exposes `exact_24h_pairs_verified` as well as first/last forecast origins.

Legacy `X,y,years` NPZ files remain loadable for historical reproducibility; older files with only `target_times` can still pass the *structural* audit but must **not** be claimed to have independently verified pair provenance. Fresh exports include both time fields, and real-data integration must check the new audit flag.

### Evidence and run status

- Source-access smoke test: [success](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45103779035).
- 72-pair 2010/2015/2020 real-data pilot: [success](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45109019800), but **development-only**, not evidence of climate change generalization.
- v1.2 sampling-correction CI: [success](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45245083243).
- The cost-bounded contiguous-year workflow is manual-only and has not been run in this development step.

### Scientific caveat

The 24-hour pair check ensures only the timestamp separation, not freedom from all leakage. In addition to chronological year splits, downstream training must prevent normalization, hyperparameter selection, seasonal climatology, target-derived covariates, and spatial overlap from leaking information across partitions. The current adapter now records observed ERA5 input coordinates rather than deriving origins from target times; the new unit test checks both source and target timestamps exist in the input dataset. A full-corpus audit must still verify raw source version and spatial/temporal metadata independently.

### Running checks

```bash
python -m pip install -e '.[test,torch,era5]'
python -m pytest
python scripts/audit_e1.py --input data/processed/e1_contiguous_dev.npz \
    --metadata data/processed/e1_contiguous_dev.json
```

Any NPZ produced before this commit must be regenerated to populate `source_times` and receive the stronger provenance check.
