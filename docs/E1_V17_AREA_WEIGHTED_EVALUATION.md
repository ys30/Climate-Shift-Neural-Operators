# E1 v1.7 — observed-latitude area-weighted ERA5 evaluation

## Motivation

Previously the 2008–2012 ERA5 development benchmark reported equal-weight grid-point MAE/RMSE/bias and a global P95 hot-gridcell diagnostic. This is not an area-weighted geographical score. At 25–50°N, regular latitude-longitude cells cover different physical areas; scientific evaluation should therefore report geographically meaningful weights.

## Implementation

- ERA5 exporters now store `latitude` from the **actual xarray input coordinate** aligned with each row of the 17×17 forecast grid. Multi-year JSON metadata adds `latitude_degrees`.
- The strict data loader validates latitude length, degree range, finite values, and unique rows.
- The structural audit cross-checks the NPZ row coordinates against JSON metadata when both are present.
- `latitude_weighted_metrics` evaluates spatial MAE, RMSE, and bias by weights proportional to cos(latitude). It averages over time and longitude with equal weights, justified for regular longitude/latitude spacing; it is not an exact cell-area map or land-masked estimate.
- `compare_e1` retains existing unweighted `metrics` for backwards comparability, adds separate `latitude_weighted_metrics` if true latitudes exist, and passes latitudes to `hot_diagnostics` to area-weight exceedance fractions and recall.
- `run_e1_multiseed` reports weighted MAE/RMSE/bias mean, across-seed SD and individual values when the source dataset contains latitude coordinates.
- Legacy datasets without latitude still run, but explicitly report `area_weighting_status: missing_latitude_legacy_unweighted`. Do **not** infer latitude from assumed geographic bounds for older data.

## Evidence

- First successful 2008–2012 real ERA5 pilot: [run 45292352961](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45292352961).
- v1.6 Unit tests: [run 45293208732](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45293208732) — passed.
- That v1.6 run **predates this change**: the published v1.6 numerical scores are **unweighted** and must remain labeled as such.
- The new v1.7 weighted scoring methods are code and tests only until a fresh actual ERA5 run is executed. Existing pilot numbers must not be rewritten or called weighted.
- The 3-seed ERA5 workflow is manual-only; no three-seed scientific results have been confirmed.

## Required next experiments

1. Confirm v1.7 CI, then run the manual three-seed ERA5 workflow on updated main and archive its data provenance and weighted/unweighted results.
2. Evaluate model-vs-persistence differences across independent seeds, **without** interpreting between-seed dispersion as confidence intervals over climate years.
3. Add held-out independent years, seasonal/grid-point exceedance thresholds, block-bootstrap uncertainty across weather systems, and more realistic spatial and future-climate shifts.

**No claim of generalization to a warmer climate is supported yet.**
