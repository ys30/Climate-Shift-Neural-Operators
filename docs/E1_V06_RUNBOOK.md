# E1 v0.6 development protocol

This implementation combines bounded multi-year WeatherBench 2/ERA5 extraction with a three-way prototype comparison. The real cloud source has **not** been downloaded or tested in the connected GitHub environment yet. No research conclusions have been established.

## Commands

```bash
python -m pip install -e '.[test,torch,era5]'
python -m pytest
python scripts/export_wb2_multiyear.py --years 2010 2015 2020 \
  --max-per-month 8 --stride 4 --output data/processed/e1_multiyear_dev.npz
python scripts/compare_e1.py --input data/processed/e1_multiyear_dev.npz \
  --train-end 2010 --val-end 2015 --test-end 2020 \
  --epochs 4 --width 8 --modes 4 --output outputs/e1_v06_dev.json
```

The years and sample counts above are **development-only**. They are not a preregistered, representative test; they contain only 8 selected input times per month and have very small data size. Results MUST NOT be interpreted as generalization evidence. The WeatherBench data may require local network access and relevant terms review. Missing months are logged, not silently counted.

## Safeguards

- Predictand: +24 h 2-m temperature only, using exact timestamps and identical input information across trained models.
- Training-target mean and standard deviation used for all sample normalization; validation used only to select best checkpoint; held-out test only scored once.
- Both FNO and U-Net use same number of epochs, optimizer, training samples, learning rate and seeded order; they do **not** have identical parameter counts, training FLOPs or matched tuning budgets.
- Output JSON includes dataset SHA-256, counts, split years, seeds, validation best MSE and parameter counts.
- Persistence baseline uses input temperature unchanged.
- No area weighting, extreme event diagnostics, observational uncertainty, multi-seed confidence intervals, hyperparameter-search parity, or future warming response implemented yet.
- This script loads the bounded extracted NPZ into RAM. **It is not a streaming large-data training pipeline.**

## Research-quality gate

Before interpreting any model skill: run real ERA5 extraction, verify variable metadata and time alignment, freeze dense contiguous year blocks (with independent validation and test periods), audit spatial sampling and seasonal representation, and run matched-budget U-Net/FNO across independent seeds. CMIP6 must then be evaluated separately.
