# E1 v1.9 — monthly and grid-local extreme threshold diagnosis

## Motivation

The verified [three-seed ERA5 run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45295324701) used a **single global P95** threshold, which can conflate summer heat and geographic climate differences. v1.9 introduces a complementary, more location- and season-aware development indicator.

## Exact definition

For each grid cell `(latitude_i, longitude_j)` and calendar month `m`, compute `q95_mij` from **training targets only**, for the corresponding month across years. Then for each held-out test target and model prediction, define the hot grid-cell classification as `T > q95_mij`.

- Source: true target timestamps in ERA5 NPZ, not inferred data order.
- Fitting source: **only 2008–2010 training target temperatures**, no 2011 validation or 2012 test observations.
- Coverage minimum: 20 training samples per calendar month; otherwise the summary reports an explicit unavailable status and does not fabricate a local P95.
- Diagnostic measures: test exceedance frequency, forecast exceedance frequency, frequency bias, recall, precision, and conditional hot-temperature bias.
- Uses **cos(latitude)** weighting when the actual ERA5 latitude row coordinates are present, otherwise reports unweighted metrics. The reported raw hot-gridcell count remains unweighted and is not an independent-event count.
- Retains the earlier single global training-P95 diagnostic in the report for comparability.
- New modules: `src/climate_shift/seasonal_extremes.py`, `tests/test_seasonal_extremes.py`; integrated through `compare_e1.py`, summarized by `run_e1_multiseed.py`.

## Limitations

This is **not** a validated heatwave definition: sampling is only 24 source-target pairs per month and may not preserve consecutive days. The threshold is estimated from three historical training years with at most 72 samples per month at each gridpoint, and confidence of the 95th percentile is limited. A future climate science benchmark requires a substantially longer continuous historical baseline, day-of-year smoothing, multi-day event rules, area weighting, independent years, and blocked weather-event uncertainty. The 2012 holdout is **not** a CMIP6 future warming experiment.

## Run evidence and version boundaries

- v1.8 CI passed: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45313983897
- Verified 3-seed ERA5 replication **ran earlier v1.6 code**, before area weighting and before monthly local thresholds.
- The existing three-seed numerical results are **unweighted global-P95** results. No local seasonal P95 or weighted ERA5 numerical results are asserted until the newer workflow has run.
- v1.9 code must pass CI before proceeding to a manual new real-data comparison.

Keep expensive GitHub Actions workloads manual-only.
