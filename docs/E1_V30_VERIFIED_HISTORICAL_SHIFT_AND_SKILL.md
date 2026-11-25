# E1 v3.0 — VERIFIED historical temperature-distribution shift vs model skill

**Successful real-data run:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45711150645  
**Archived JSON and source audit:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45711150645/artifacts/15067921629  
**Run commit:** `b487571b3a778d3c223044c8afa726230e7dcc0d`.

The original workflow run 45708056678 failed **before extracting data**, because the data-only workflow ran an unrelated torch-dependent test without installing torch. The replacement workflow restricted tests to `tests/test_historical_shift.py`, and every stage—including real ERA5 export, coverage audit, training-only climatology, year anomalies and archiving—passed.

## Reproducible setup

Western US WeatherBench2 ERA5 2-m temperature, 17×17 grid, +24h paired timestamps, 24 selected pairs/month/year. The temperature reference (month-by-local-gridcell mean and standard deviation) uses **only the targets in 2008–2010**; 2011 and 2012–2014 cannot affect climatology estimation. All gridcell statistics weight rows by cos(latitude). This is a descriptive sample-period distribution comparison—not anthropogenic warming attribution, future climate projections, or robust high-frequency climatology.

## Verified shift audit

| Year | Area-weighted month-local temperature anomaly (K) | Mean standardized anomaly | Fraction above training month-local +2σ | Fraction below −2σ |
| --- | ---: | ---: | ---: | ---: |
| 2008 | −0.075601 | −0.059914 | 0.011371 | 0.026203 |
| 2009 | +0.126357 | +0.072563 | 0.018263 | 0.026350 |
| 2010 | −0.050756 | −0.012649 | 0.011118 | 0.025941 |
| 2012 | +1.196836 | +0.240760 | 0.033191 | 0.020069 |
| 2013 | +0.174828 | +0.074905 | 0.022477 | 0.038325 |
| 2014 | +1.015622 | +0.479481 | 0.087384 | 0.032445 |

**Crucial distinction:** 2014 has a lower average month-local temperature anomaly than 2012, but a greater positive standardized anomaly and >+2σ fraction. The mean anomaly and upper-tail exceedance distribution reflect different dimensions of historical sample shift.

## Join to previously verified 100-epoch, three-seed prediction metrics

Prediction evidence: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45679648002  
Prediction artifact: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45679648002/artifacts/15052934806

| Heldout year | Above +2σ fraction | FNO weighted RMSE (K) | U-Net weighted RMSE (K) | Persistence local P95 recall | FNO local P95 recall | U-Net local P95 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2012 | 0.033191 | 2.847327 | 2.809589 | 0.542995 | 0.442778 | 0.419875 |
| 2013 | 0.022477 | 2.768649 | 2.775122 | 0.516313 | 0.389761 | 0.317681 |
| 2014 | 0.087384 | 2.956731 | 2.934970 | 0.687260 | 0.592163 | 0.542644 |

### What is supported

- Both trained architectures have lower weighted mean forecast error than Persistence and lower local training-P95 gridcell recall in **each of these three years**.
- 2014 is the year with the greatest +2σ exceedance fraction and has the highest FNO weighted RMSE, **but also the highest** FNO local P95 recall.
- The crude direction of the year-level association between shift indicators and recall is **not** a direct measure of model sensitivity to shift, because the event prevalence and underlying weather difficulty vary simultaneously.
- **n=3 meteorological test years:** no inferential correlation, p-value, dose–response trend, or attribution claim is justified. Repeated gridcell/sampled date observations are not independent years.
- The +2σ indicator and the training-month-local P95 Recall denominator are distinct event definitions; do not interpret them as interchangeable probability thresholds.

### Recommended next discriminating study

Rather than begin CMIP6 solely on the basis of three historical years, add **paired conditional error decomposition** by heldout-year, month and training-standardized anomaly bins, for both FNO/U-Net and Persistence, and stratify high-tail precision/recall by event prevalence. Estimate uncertainty by **year or weather-block bootstrap**, not treating neighboring gridcell measurements as iid. Maintain a frozen, training-only reference. Then add climate-shift experiments with historical-to-SSP benchmark with matching predictors and cautious bias correction.
