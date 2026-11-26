# E1 v5.14 — Third independent GCM feasibility audit

## Context: actual v5.13 outcomes
The verified [v5.13 workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45894892093) reports four of four GCM×SSP groups for each: standard FNO RMSE beating its own Persistence, 4× FNO tail-weight increasing Recall versus 1×, 4× weight worsening RMSE, and Persistence hot Recall exceeding 4× FNO. These four scenario cells come from only **two** GCMs, not four independent physical model replications. No significance or broad causal claim.

## Why not choose ACCESS-CM2 automatically
The previously verified [v4.2 time coverage audit](E1_V42_CMIP6_SMALL_SLICE_CALENDAR_GATE.md) found catalog-selected ACCESS-CM2 SSP585 spans **2201–2300**, not 2015–2100. It cannot be used for common 2030–2080 comparisons without a different real verified store.

## v5.14 preflight
Re-download the actual public Pangeo CMIP6 consolidated metadata catalog and compute SHA256; pair original model/member/grid Historical/SSP245/SSP585 daily `tas`. Predeclare candidate priorities CAMS-CSM1-0, CESM2, CESM2-WACCM, CMCC-CM2-SR5; pilot at most two genuinely catalog-returned distinct models.

Read native per-store CF-calendar endpoints, Kelvin schema, grid spatial dimensions and bounded tiny regional source samples using existing audited source reader. Require a historical source reaching 1980–2014 and both SSP scenarios covering 2015–2080 on a compatible calendar and grid shape. Record candidate-level pass and fail reasons **rather than suppressing unusable data**.

This is a first eligibility gate ONLY. After success, separately verify actual Western-US regional grid coordinate geometry and source/target +1 day dates. Then independently train new Historical-only GCM native-grid predictors; NEVER present AWI/BCC weights as a no-regrid directly transferable third-GCM model. A third GCM is a stronger structural replication, but additional GCM members and continuous multi-year samples are necessary for inference about future climate sensitivity and heatwave persistence.

## Provenance rules
No fabricated store URL or model inventory; no SSP parameters fitted or selected. Archive source URL, catalog SHA, calendar, date ranges, native geometry and tiny Kelvin evidence.
