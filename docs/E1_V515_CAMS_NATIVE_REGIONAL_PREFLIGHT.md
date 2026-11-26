# E1 v5.15 — CAMS-CSM1-0 native 25–50°N, 125–100°W regional data gate

The previous [v5.14 catalog and tiny-field audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45895659921) passed CAMS-CSM1-0 and rejected the selected CESM2 Historical candidate for the required 1980–2014 span. This does NOT establish third-GCM trained forecasting performance.

## Exact next gate
- Reuse archived real v5.14 CAMS member/grid/source URLs, compare against freshly retrieved original CMIP6 catalog SHA and store paths. Never invent any third-GCM URL.
- Read no more than two paired daily **regional** Kelvin grids from Historical (2010 Jan 15 and July 15), SSP245 and SSP585 (2030 Jan 15 and 2050 July 15), with source-target exactly +1 native day.
- Verify native regional shape has 4–2000 grid cells, latitude/longitude coordinate SHA and shape match across Historical and SSP simulations, calendar identical, finite physical Kelvin source and target, and native persistence RMSE sanity scores.
- Record detailed source locations and provenance, but no actual whole-year regional array nor performance/training result.

## Subsequent workflow
If successful, export a fixed 1980–2004 historical train and 2005–2014 validation regional cohort for CAMS independently, fit its own normalizer and month-local P95, train a third native-grid FNO/U-Net suite using historical-only checkpoint selection, then test SSP with unchanged parameters. Do not load AWI or BCC native-grid weights as allegedly transferable to CAMS. Explicitly label any additional future years/dates as exploratory design developed after the earlier SSP experiments.

## Limitations
Two source/target dates per experiment cannot establish complete historical climate coverage, SSP future truth, prediction skill, heatwave duration or forced climate response. Multiple SSPs for one GCM share model structural uncertainty, so do not call two SSPs two independent GCM replications.
