# E1 v1.0 actual ERA5 pilot results — audited development evidence

**Run:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45109019800
**Commit:** f8835480d3c067cc9509ea22d2c3e91a3c486f1f
**Outcome:** completed / success (all extraction, audit, and comparison steps succeeded).
**Artifact:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45109019800/artifacts/14788757932
**Date (UTC):** 2026-11-23
**Scientific classification:** integration / development test, NOT full historical-OOD evidence.

## Data provenance

- Source: WeatherBench 2 ERA5, `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`
- Variable: 2m_temperature, Kelvin; +24-hour target matching.
- Region: 25–50N, 125–100W; 17 × 17 selected grid points at 1.5°.
- 72 source-target pairs, precisely 24 for 2010 training, 24 for 2015 validation, 24 for 2020 testing (2 sampled source times per month); no skipped months.
- First target 2010-01-02 00:00 UTC; last target 2020-12-03 00:00 UTC.
- NPZ SHA-256: `1640ba09ad25e2109664121516fd17a719dc27813a0ba90330cebc1fd314a2b9`.
- The NPZ provenance audit verified timestamps, year/month counts and source metadata checksum.

## Actual model metrics (physical unit: K)

| Model | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: |
| Persistence | 2.398760 | 3.605571 | +0.066705 |
| U-Net | 5.869187 | 7.355902 | -1.230616 |
| FNO | 9.716702 | 11.665550 | -4.478026 |

Both trainable networks received only **2 training epochs**, an extremely small amount of training data, and no search over matched compute budgets. These outcomes are **not a valid architectural ranking**.

Development P95 training-reference threshold: 305.347595 K (a global grid-cell threshold, not a seasonally adjusted heatwave definition).

| Metric | Persistence | U-Net | FNO |
| --- | ---: | ---: | ---: |
| Predicted hot fraction | 0.071799 | 0 | 0 |
| Reference hot fraction | 0.076125 | 0.076125 | 0.076125 |
| Recall | 0.759470 | 0 | 0 |
| Hot-conditional bias (K) | -0.634410 | -12.754809 | -21.663336 |

## Interpretation boundaries

1. This proves remote ERA5 access, small data extraction, structural data audit, three-model runtime and metric output on real values.
2. It does NOT prove future warming OOD generalization, climate trend conservation, physical consistency, fair operator-vs-CNN comparison, or statistically reliable extreme event detection.
3. Dense year blocks, more training, multiple seeds, truly independent spatial/temporal splits, domain-specific seasonality and area weighting are necessary before any publication claim.
4. Do not transfer these model-comparison numbers into the manuscript abstract/conclusions or industry portfolio as scientific performance rankings.

## Next experiment

Run the manually dispatched, runtime-limited `e1-contiguous-dev.yml` workflow using consecutive full calendar years, daily source times (up to 24 per month), separate train/validation/test years and improved training. Inspect source coverage and results before escalating to full-scale runs.
