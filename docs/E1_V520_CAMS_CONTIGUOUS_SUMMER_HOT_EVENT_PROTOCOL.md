# E1 v5.20 — Contiguous 2050 CAMS summer daily-mean hot-event pilot

## Established evidence before new event experiment
[Three-GCM frozen SSP run 45908305246](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45908305246) succeeded. In all 3 GCMs and both SSP245/SSP585 scenarios, standard FNO RMSE beat own Persistence; FNO 4× training hot weight raised daily P95 Recall relative to FNO1× while slightly worsening weighted RMSE; Persistence daily-hot Recall still exceeded FNO4×. These are descriptive results with just 3 independent GCMs and 72 dates per SSP case.

## Next question
Sparse dates cannot establish multi-day duration. This initial bounded **continuous** pilot fixes CAMS source dates 2050 June 1 through August 30 inclusive (91 dates) and one-day targets June 2 through August 31 inclusive. Read only **92** contiguous source-region native daily Kelvin grids per SSP for 2 SSPs. Do not pull entire global fields or claim full-year completeness.

## Historical-only model and threshold gate
Recover previously successful CAMS Historical cohort v5.16 and three 100-epoch Historical-only weight arms v5.17. Require full SHA256 and provenance checks for all 18 checkpoints, independent CAMS mean and SD and historical-only monthly local TAS P95 BEFORE opening any SSP scenario Zarr. No refitting, parameter changes or scenario-based checkpoint selection.

## Event definition and metrics
For each grid cell and each target day, mark a simulated high temperature if native daily mean `tas` exceeds that cell's **historical training-target local P95 for the target's month**. An event consists of at least 3 **consecutive** target days exceeding that threshold at the SAME grid cell. Mark all days within each qualifying event. Compare model-predicted vs GCM-simulated reference event days at the same cell and date using cosine latitude weights. Report:
- area-weighted event-day Recall = spatial-temporal weighted intersection / observed qualifying hot event days;
- area-weighted event-day Precision = weighted intersection / predicted event days;
- count of 3-consecutive-day pixel windows; and
- native Persistence and all fixed FNO/U-Net seeds/hot weights daily RMSE, daily P95 Recall and Precision.

Do not report 1.0 Recall if source has zero qualifying events; use explicit null instead. The event condition is **daily-mean TAS**, NOT `tasmax` heatwave exposure or human heat stress.

## Scientific restrictions
This is an exploratory one-year, one-GCM/one-member **summer window** (newly designed after previous SSP scoring), not multi-year or fully blind GCM replication. Events are left- and right-censored at June 2 and August 31 because outside-window sequence is not included. It evaluates pixel-local event-day hit rate, not synoptic heatwave system tracking or object-based event matching. Warm-mean scenario fields are simulations, not observed future data. The next preregistered expansion should use multiple contiguous years, an additional GCM, and event-duration distribution with interval-censoring handling.
