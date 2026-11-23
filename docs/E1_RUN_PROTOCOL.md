# E1 training and evaluation — version 0.5

**Current status:** Development baseline; not a peer-reviewed result, real ERA5 experiment, heatwave analysis or climate-shift experiment.

The 2019 seven-day extraction in the WeatherBench 2 source card is a data-access smoke test only. It cannot be split into independent multi-year train/validation/test sets. To run the trainer, first prepare a multi-year NPZ with identical `X`, `y`, `years` arrays, where `years` represent **target** years, and document source/lead alignment.

```bash
python -m pip install -e '.[test,torch,era5]'
python -m pytest
python scripts/train_e1.py --input data/processed/multi_year_pairs.npz \
  --train-end 2015 --val-end 2018 --test-end 2022 \
  --embargo-years 1 --epochs 10 --output outputs/e1_baseline.json
```

The years above are **illustrative only**; they do not establish a frozen scientific split. The command will reject missing partitions.

## Scientific controls

1. FNO receives the same single temperature predictor as persistence; the target is exactly 24 h later **only if** upstream extraction followed the documented adapter.
2. Normalize both predictor and target using **training-target** mean and standard deviation. Validation and test values are not used to fit normalization.
3. Select the model checkpoint on validation MSE. Compute test metrics only after checkpoint selection.
4. Include input checksum, seed, train/validation/test sample counts, configuration, and diagnostic JSON.
5. Do not compare models on an artificially short or unrepresentative time span, and do not interpret point forecast skill as climate-change response fidelity.
6. Current per-grid averages are unweighted. Latitude-area weights, heatwave duration, seasonality, spatial holdouts, and block bootstrap remain future work.
7. The current predictor is only 2-m temperature. This is not yet an information-matched competitive meteorological forecasting benchmark.

## Required next gate

Verify actual multi-year ERA5 availability and licensing, inspect full-grid shapes/time coordinates, pre-register model/region/years before a final held-out evaluation, and complete a real-data sample run with traceable logs.
