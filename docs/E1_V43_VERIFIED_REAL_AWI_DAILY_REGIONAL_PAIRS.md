# E1 v4.3 — VERIFIED actual AWI CMIP6 daily +1-day Western-US regional temperature pairs

**Successful end-to-end source audit**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764223116  
**Original small JSON artifact**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764223116/artifacts/15091098317

The successful GitHub workflow installed and passed strict native-calendar, regional-coordinate and ensemble-ID unit tests, read actual small regional temperature input/target grids, checked source-to-target daily lead, and archived only derived JSON.

## Actual native CMIP6 grid and provenance

- Model: **AWI-CM-1-1-MR**, ensemble **r1i1p1f1**, native grid `gn`; `day/tas` temperature in **K**, all experiments calendar **proleptic_gregorian**.
- Western US requested region: **25–50°N, 125–100°W** (`235–260°E`) without regridding.
- Actual native selected coordinates: **25.714192641°–49.090686051°N** and **235.3125°–259.6875°E**, giving **26×27 grid cells (702 per daily frame)**. These are **not** ERA5's 17×17 grid.
- Every experiment returned two dated pairs with `tas(t)` exactly followed by `tas(t+1 native day)`, source/target same shape, **no nonfinite values in the sampled regions**, and absolute temperature in a physically plausible Kelvin range.
- Raw temperatures are not archived; the derived JSON records exact source/target dates and descriptive consistency diagnostics.

## Verified bounded *pilot only*: simple Persistence one-day RMSE

These are only **one individual pair per row**, 702 spatial cells, and **NOT** reliable forecast performance summaries for future climate.

| Experiment | Input source day | Target next day | One-pair regional cos(latitude)-weighted Persistence RMSE (K) |
| --- | --- | --- | ---: |
| Historical | 2010-01-15 | 2010-01-16 | 1.544 |
| Historical | 2010-07-15 | 2010-07-16 | 1.513 |
| SSP245 | 2030-01-15 | 2030-01-16 | 2.297 |
| SSP245 | 2050-07-15 | 2050-07-16 | 1.776 |
| SSP585 | 2030-01-15 | 2030-01-16 | 1.550 |
| SSP585 | 2050-07-15 | 2050-07-16 | 1.492 |

The pilot excludes any entire SSP domain-shift learning or forecasting comparison; weather-day difficulty is not held constant across projected scenarios, and no neural networks were trained or tested on SSP. Do **not** compare these sparse single-day values as evidence that SSP245 or SSP585 is easier or harder.

## What is now verified

- Catalog identities and actual 2015–2100 dates of the AWI Historical, SSP245 and SSP585 trio.
- Correct source-variable identity and Kelvin native daily-mean target semantics.
- One compact native-coordinate Western-US region that is identical in **shape** for all scenarios.
- Exact +1 native calendar day source/target formation on *six* predeclared source dates.
- Finite and plausible physical temperature values on each sampled 26×27 frame.
- A nonzero, calculable Persistence baseline as a **pipeline unit/sanity check**, not a generalizable skill score.

## Still required before future-neural-operator results

1. A truly representative **multi-year** daily sample from the same time/region: Historical train/validation period, distinct SSP245/SSP585 future periods. Verify every selected source–target day with native CF calendar and no SSP→Historical boundary transitions.
2. Check complete selected training/validation/test dataset coverage, spatial missingness, original model timestamps, sample-vs-continuous interpretation, region grid *coordinate values* and source-store version for reproducibility.
3. Fit local monthly means, SD and P95 **only on historical training daily targets**; do not use SSP distributions for preprocessing, checkpoint selection or weight tuning.
4. Prespecify daily model and baselines (FNO, U-Net, Persistence), training budget and loss candidates before SSP model evaluation. Hold out future SSP scenarios entirely from model fitting.
5. Report independent climate-model/member effects and climate-change-signal fidelity separately; a single AWI member, without multi-model confirmation, does not establish climate-system ensemble robustness.

See the [frozen v4.3 regional protocol](E1_V43_AWI_CMIP6_DAILY_PAIR_REGIONAL_PROTOCOL.md) for prospective Historical split 1980–2004 training and 2005–2014 validation. The true future evaluation has **not** been executed.
