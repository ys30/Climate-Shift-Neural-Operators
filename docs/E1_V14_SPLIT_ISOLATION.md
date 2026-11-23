# E1 v1.4 — chronological fold boundary isolation

## What changed

E1 previously assigned training, validation and test periods by **target year**, and checked that each individual input-target pair had a +24h offset. That is necessary but insufficient for a conservative between-fold independence guarantee.

The new `src/climate_shift/split_isolation.py` reads the **observed** `source_times` and `target_times` from the NPZ, rejects invalid offsets, empty or overlapping masks, and demands that the maximum target timestamp of an earlier fold is strictly before the minimum input timestamp of the next fold. It tests train-validation, validation-test, and train-test boundaries.

Example: if training contains a target at 2010-12-31 00:00 and validation includes an input beginning at that same instant for a 2011-01-01 00:00 target, this strict guard rejects the dataset. An appropriate temporal buffer, or removing boundary samples, is needed. This policy is deliberately conservative; using a previous observed temperature as a forecast input is not necessarily illicit in all operational forecasting designs, but here we avoid sharing a raw timestamp between the training labels and validation inputs.

Both `train_e1` and `compare_e1` write `split_isolation` diagnostic metadata into their JSON reports. Legacy NPZ files without explicit source and target time coordinates are reported as `unverified_legacy_no_pair_times` for backward-compatible development tests. They cannot be treated as scientifically audited. The manually invoked contiguous-year ERA5 experiment now uses `--require-pair-times` and fails rather than silently accepting missing timestamp provenance.

## Verification

- E1 v1.3 final CI passed: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45273471343
- E1 v1.0 real ERA5 72-pair development run passed: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45109019800
- New v1.4 regression tests: `tests/test_split_isolation.py`. Final v1.4 CI should be checked before marking this revision verified.

## Scientific limits

This is a chronological *timestamp isolation* guard, not proof of statistical independence. Nearby atmospheric states remain temporally and spatially correlated. Full E1 needs sufficiently separated rolling-origin evaluations, independent years, area weighting, multi-seed matched-compute baselines, blocked confidence intervals and model/data audits. Sparse monthly samples must not be used to estimate continuous heatwave duration.

## Manual benchmark

```bash
python scripts/compare_e1.py --input data/processed/e1_contiguous_dev.npz \
  --train-end 2010 --val-end 2011 --test-end 2012 \
  --require-pair-times --epochs 5 \
  --output outputs/e1_contiguous_dev_metrics.json
```

The data extraction and training workflow is manual-only to avoid unsolicited compute charges. No contiguous-year metrics exist until it is explicitly run and its artifacts inspected.
