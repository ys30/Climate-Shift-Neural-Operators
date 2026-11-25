# E1 v2.9 — verified 2012, 2013 and 2014 independently scored ERA5 holdouts

## Evidence

- **Successful automated workflow:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45679648002
- **Run SHA:** `8a8b28127c537e3fdb1cee48e32477818d433fe5`.
- **Diagnostic artifact:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45679648002/artifacts/15052934806
- ERA5 2-m temperature, 17×17 western-US grid, exactly +24h paired timestamps. Train 2008–2010, validate 2011, independently assess 2012–2014, 288 sampled pairs per heldout year. Three independent training random seeds 11, 22, 33, 100 epochs, validation-selected checkpoint; no tuning on any test year.

## Observed year-specific results

**Three-seed averages.** RMSE computed using actual cos(latitude) grid-row area weights, and high-tail recall uses training-only monthly/local-gridcell P95.

| Test year | Model | Weighted RMSE (K) | Weighted monthly grid-local P95 recall |
| --- | --- | ---: | ---: |
| 2012 | Persistence | 3.461329 | **0.542995** |
| 2012 | FNO | 2.847327 | 0.442778 |
| 2012 | U-Net | **2.809589** | 0.419875 |
| 2013 | Persistence | 3.400217 | **0.516313** |
| 2013 | FNO | **2.768649** | 0.389761 |
| 2013 | U-Net | 2.775122 | 0.317681 |
| 2014 | Persistence | 3.671990 | **0.687260** |
| 2014 | FNO | 2.956731 | 0.592163 |
| 2014 | U-Net | **2.934970** | 0.542644 |

Across these three heldout years, each neural model improves area-weighted RMSE over Persistence but reduces the rate of detecting hot gridcell exceedances using local month-specific training P95 thresholds.

**Critical interpretation:**
- Each row is based on the same trained seed models and sample selection, so *three test years are more informative than a single 2012 aggregate*, but they do not represent a statistically broad climate sample.
- A stronger result than pooled metrics is the common direction of the mean-error vs hot-gridcell-recall tradeoff in **all three** years.
- 2013 U-Net hot recall 0.317681 vs Persistence 0.516313 is a notable shortfall, not yet a mechanism attribution.
- U-Net has slightly lower RMSE in 2012 and 2014; FNO is marginally better in 2013. No general architecture ranking is established.
- These are discrete 24-pair-per-month gridcell exceedance classifications, not independently sampled heatwaves or extreme duration events.
- Historical year-to-year distribution changes are **not direct evidence of a future climate-change signal or of CMIP6 warming generalization**.

## Next stage (v3.0)

Quantify heldout year-by-year temperature anomalies relative to **training-only monthly grid-local climatology**, with latitude weighting. Separate descriptive historical OOD signatures from causal attribution. After that, controlled CMIP6 historical-to-SSP scenario comparisons can test actual future-warming response. 
