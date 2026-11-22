# Data source card: WeatherBench 2 ERA5 6-hourly (E1 candidate)

**Status:** Source URL verified from official WeatherBench 2 documentation. Real-data download, licensing inspection and first forecast run are NOT complete.

Official guide: https://weatherbench2.readthedocs.io/en/latest/data-guide.html

Selected public cloud store:
`gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`

Official documentation states 1959–2023 and 6-hourly ERA5 variants and lists 240×121 (1.5°) as the benchmark-paper evaluation resolution. Individual datasets can have distinct license terms: inspect the applicable bucket LICENSE **before redistribution or commercial use**.

## First bounded extraction

```bash
pip install -e '.[test,torch,era5]'
python -m pytest
python scripts/export_wb2_pairs.py --start 2019-07-01 --end 2019-07-07 --max-samples 16 --output data/processed/wb2_e1_24h.npz
```

Adapter makes exact +24h pairs from existing 6-hourly timestamps, restricted to lat 25–50 N, lon 125–100 W. Only `2m_temperature` is used in this first adapter. It writes an NPZ containing `X`, `y`, `years` and a separate metadata JSON with the source, time bounds, grid subset, reported source unit and SHA-256 checksum. The prediction target and predictor have **the same variable** but different times.

The example selection is only a small development subset. 2019 is not a preregistered test year and must not be used to claim independent out-of-distribution climate results. The current adapter is deliberately not a full dataset extraction pipeline.

**Remaining:** verify bucket terms and model input metadata, freeze multi-year splits, temporal boundary rules, source meteorological variables, normalize from training only, build streaming/batched real-data access, run full U-Net/FNO comparisons, and document uncertainty/area weighting.
