# E1 v5.16 — CAMS-CSM1-0 independent Historical-only native training cohort

## Verified prerequisite
v5.15 repaired GitHub run 45898911977 passed actual CAMS regional Kelvin +1 day pairs and fixed native-coordinate fingerprints for Historical, SSP245 and SSP585. This does NOT yet imply independent trained model skill.

## Immutable export contract
Historical native CAMS only, using the source and native coordinates verified and archived in v5.15. Do not open any SSP model fields. Training 1980–2004: days 6/16/26 monthly (900 +1 day pairs). Validation 2005–2014: days 8/20 monthly (240 pairs). Exactly 1140 pairs; 75 training targets and 20 validation targets per calendar month. Preserve native CF numeric date and original cftime strings, not Gregorian coercion.

Read native grid latitudes 25–50°N, longitudes 235–260°E only, and enforce SHA256 equality with independent v5.15 audited coordinate values. The shape is **read from the verified CAMS source**, not copied from AWI (26×27) or BCC (23×23). Reject missing fields, Kelvin values outside 150–350, source-target time difference other than 1 native CF day, altered calendar or train/validation contamination.

Derive monthly local P95 solely from CAMS Historical *training* targets. Do not reuse AWI/BCC thresholds, means, scales, model weights or checkpoints. Record historical-validation Persistence RMSE as a diagnostic, not as evidence of future SSP skill.

## Output
Immutable `e1_cmip6_v516_cams_historical.npz` plus JSON provenance with source audit and SHA, grid, calendar, counts, year/days schedule, 12 monthly P95 target counts, observed historical Persistence and explicit `ssp_data_opened:false`.

## Interpretation
This is a **sparse historical training cohort**, not a continuous climatological series. v5.17 should independently fit CAMS-only normalizers and architectures using Historical data, then freeze 18 checkpoints before reading any SSP targets. Additional GCMs and full multi-year daily histories remain essential for general climate-signal claims.
