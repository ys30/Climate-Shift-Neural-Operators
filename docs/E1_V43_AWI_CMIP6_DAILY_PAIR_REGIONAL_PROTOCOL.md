# E1 v4.3 — CMIP6 AWI daily-pair regional extraction design (locked before model training)

## Evidence gate and frozen source

- Native Zarr opening was verified for AWI-CM-1-1-MR member `r1i1p1f1`, grid `gn`, on historical, ssp245 and ssp585 (v4.1 success: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759128555).
- Real native CF dates and Kelvin data samples from [v4.2 archive](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759757258/artifacts/15089577174) show **all AWI datasets cover historical 1850–2014 and both SSP projections 2015–2100**. The erroneous ACCESS-CM2 SSP585 store starts 2201. A corrected automated common-period gate is being run separately.
- AWI native grid shape `192 lat × 384 lon`, approximately latitude −89.284 to +89.284 and longitude 0 to 359.0625; extract Western US `25°–50°N, 235°–260°E` (= 125°–100°W) without extrapolation. The source grid is **not** the ERA5 17×17 grid.
- CMIP6 `day/tas` is daily mean **absolute temperature** in Kelvin; the corresponding study defines **daily `tas(t)` → `tas(t+1 day)`** for each model/source/scenario. It is not equal to the existing instantaneous ERA5 +24h target. All future skill results should be labelled a separate daily native-model forecast evaluation.

## Next bounded physical-data feasibility gate

Select a fixed small, sparse **regional** set of days from AWI Historical and both SSPs (e.g., winter and summer dates within available ranges). For each selected date:
1. Determine time index using the source **native CF calendar** and verify the adjacent time coordinate is exactly +1 native day, not `datetime64`-forced Gregorian arithmetic.
2. Select only the region via coordinate-derived spatial indices, not hard-coded ERA5 grid dimensions. Longitudes normalize to 0–360; no arbitrary interpolation at this audit stage.
3. Read only two *regional* temperature frames for each sampled pair; check Kelvin units, finite grid-cell fraction, source/target exact same shape, physically plausible values and no missing cells in the selected training/evaluation window.
4. Record exact timestamp strings, native grid size, coordinate bounds, missing counts, source Zarr path, source version and variable metadata in JSON. **Do not** commit or upload raw CMIP6 copyrighted/large arrays.
5. Verify historical/SSP paired dates never cross scenario boundaries and that the projection years are distinct from the Historical training and validation periods.

## Frozen prospective split for later training (not yet executed)

- Historical (new CMIP6 model) training: **1980–2004**; Historical validation: **2005–2014**.
- Scenario zero-shot tests (new CMIP6 predictor trained on Historical only): **SSP245 and SSP585 2015–2040**, plus separately **2041–2070** and **2071–2100**, if coverage, budget and daily frequency pass audits.
- Candidate architectures: FNO, U-Net, Persistence. Candidate training hot weights: **1×, 2×, 4×** (i.e. parameters 0/1/3). These are historical ERA5 candidates and are NOT expected to be universally optimal on CMIP6.
- Use the Historical **training-only** local/monthly P95 thresholds, train-only climatology, same grid and lead; select checkpoints on **2005–2014 Historical validation only**.
- Crucially do not choose loss weights, model sizes, training epochs or preprocessing after seeing SSP test metrics.
- Report mean RMSE/MAE/bias, above-P95 threshold Recall/Precision/false-positive fraction, >+2σ hot-condition weighted RMSE and bias; report projection-consistent changes rather than claim observed future outcomes.
- Do not interpret model-to-model differences as climate-ensemble independence or claim SSP ground-truth forecast transfer without sufficient independent model-members/years.

This is a preregistered design, **not an executed CMIP6 neural-operator training benchmark**.
