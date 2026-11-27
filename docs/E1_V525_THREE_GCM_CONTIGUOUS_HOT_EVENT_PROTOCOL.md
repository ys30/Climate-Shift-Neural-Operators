# E1 v5.25 — Three independent GCMs, 2050 contiguous summer heat-event diagnostic

## Data provenance
Successful immutable original archives: AWI v5.24 run 45937473743, BCC v5.23 run 45928444026, CAMS v5.20 run 45914481554. One original model member and 18 original Historical-only neural network checkpoints per GCM, 54 total. The new audit does not open climate source fields or refit any models.

## Testable comparison
Three independently modeled native regional grids each use 91 consecutive source dates from June 1–August 30 2050 and 91 day+1 target dates from June 2–August 31, in SSP245 and SSP585. Daily-mean `tas` local hot day exceeds that model's Historical training-target month-local P95. A same-pixel event requires >=3 consecutive exceeding target days. Aggregate day-and-cell matched event Recall/Precision for native Persistence, FNO and U-Net, all 3 fixed tail weights and 3 initialization seeds. Never pool absolute Kelvin RMSE across GCMs without a GCM-specific baseline.

## Interpretation boundaries
Six GCM×SSP cells represent only three independent GCMs, not six independent climate ensembles. Other biases remain: one 2050 summer, one GCM member each, window-edge censoring, simulated rather than observed future tas, no tasmax, pixel-level day matches rather than tracked heatwave objects, spatial/temporal dependence, and exploratory analysis chosen after sparse-SSP results. No inferential p-values or independent seed-ensemble claims. Preserve counterexamples rather than omit any.
