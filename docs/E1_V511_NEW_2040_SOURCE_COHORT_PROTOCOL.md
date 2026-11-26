# E1 v5.11 — predeclared new 2040 native CMIP6 sampled-source pilot

## Purpose
v5.5–v5.10 used 2030/2050/2080 days 8/20 monthly. v5.11 opens a **new year, 2040**, and **new, nonoverlapping days**, 3/7/11/15/19/23/27 monthly, first as a physical source-only extraction gate. Because this year and sample schedule are being introduced after earlier SSP exploration, this is a new documented test cohort but cannot be called a wholly pristine blind evaluation of all analytical design decisions.

## Data lineage
The same audited `day/tas` Kelvin CMIP6 stores: AWI-CM-1-1-MR r1i1p1f1 gn, proleptic Gregorian, regional native 26×27; BCC-CSM2-MR r1i1p1f1 gn, noleap, native 23×23. No regridding or cross-model pooling. Spatial 25–50°N and 125–100°W. Both SSP245 and SSP585.

## Sample contract
2 GCM × 2 SSP × 1 new year × 84 source dates = **336** native +1 day source/target pairs. Exactly 7 days/month, never the original 8th/20th. Reuse the same calendar-day schedule for each SSP. Verify native +1 day, coordinates, grid shapes, Kelvin 150–350 bounds and finite fields. Record latitude-weighted mean of source and target for each sampled date, spatial target P95 per date, plus model-specific reference grid coordinate hashes.

## Strict interpretation
This is **not** complete 2040 annual daily data: 84 dates/year still contains considerable sampling and internal weather variability. Not a climate ensemble, independent future observed truth, neural forecast result, or heatwave duration test. It is a source-access and physical-sample expansion gate, not model generalization proof.

## Next gated stage
Once output validates, apply **all frozen historical** AWI and BCC checkpoint inventories without SSP retraining or scenario-based selection to the new 2040 source/target fields and archive per-date predictions. Only after this may new-year model fidelity be compared. A separate sustained daily multi-year cohort and GCM-member expansion remain necessary for any long-term climate-signal claims.
