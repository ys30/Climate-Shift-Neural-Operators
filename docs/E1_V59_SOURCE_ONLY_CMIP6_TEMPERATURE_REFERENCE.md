# E1 v5.9 — Native CMIP6 source-only temperature reference

## Research purpose
Independent audit of *simulated CMIP6 daily mean surface air temperature*, separate from frozen trained neural forecasts. This is a physically interpretable Kelvin baseline for later climate-shift fidelity work.

## Sources, dates and controls
- AWI-CM-1-1-MR (26×27, proleptic Gregorian), BCC-CSM2-MR (23×23, noleap); each independently audited with source SHA.
- Historical reference: 2005–2014, 240 sparse target dates per GCM (9th and 21st of each month, 20 observations per calendar month).
- Projected SSP245 and SSP585: sampled 2030/2050/2080, 24 target dates per sampled year, corresponding to source 8th and 20th of each month.
- Model training used Historical 1980–2004, but its checkpoint selection used the 2005–2014 Historical validation cohort. Thus **this reference is not an independent unseen Historical benchmark**. Only simulated temperature fields, never trained predictor outputs, are used in this source-only audit.
- Verify saved source files by SHA, native source calendar, matched grid latitude/longitude coordinates, finite Kelvin values, and complete sample dates; never pool unrelated GCM native pixels.

## Metrics
Regional mean is computed as a native-grid cosine(latitude) weighted average of the mean target Kelvin field. For each SSP/year, report the difference relative to the same GCM Historical reference, and the 5th/50th/95th spatial percentiles of gridcell target temperature change.

Factorial: 2 GCM × 2 SSP × 3 sampled years = 12 source-only rows. No neural model loading, checkpoint selection, SSP fitting, or inferential p-values.

## Limitation
**Not** full annual/multi-decadal climate means, radiatively forced warming, regional projection truth, or proof of climate-response preservation. Sparse samples contain internal weather variability. A future multi-year/daily continuous benchmark and additional members are required for any claim about forced warming or genuine future extreme-event persistence.

## Next preregistered design
Plan independent densely sampled source simulations and frozen-parameter forecast runs; compare model-predicted Historical-to-future changes and spatial pattern against actual GCM-simulated target changes using the *same* time schedule and native grid, and quantify source-GCM/member uncertainty.
