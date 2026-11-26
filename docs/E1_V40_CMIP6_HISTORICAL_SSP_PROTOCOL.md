# E1 v4.0 — CMIP6 climate-domain shift: data compatibility and preregistered protocol

## Aim

Extend historical ERA5 +24h generalization and tail-weighted FNO/U-Net results into a real **Historical-to-future SSP domain-shift** evaluation. The existence of a CMIP6 data store or a metadata match is **not evidence of evaluated future-climate skill**.

## Two distinct experiments; DO NOT conflate target semantics

**Track A: native CMIP6 daily-tas, climate-model within-member transfer (preferred first).**
Use a single `source_id / member_id / grid_label` for historical, ssp245 and ssp585 with **table_id=day, variable_id=tas**. Require matching daily means, daily lead and native calendar handling. Train FNO/U-Net and Persistence on a *new CMIP6 daily-mean paired* Historical cohort and validate on a disjoint Historical period, testing SSP data separately. This tests source-model projected distribution transfer but **does not directly transfer existing ERA5 +24h snapshot predictor weights**. Report full grid-cell anomaly shift, per-scenario RMSE, bias, local hot recall AND precision, conditional hot-bin RMSE; evaluate retained-warming magnitude with a training-only reference and report inability to infer ground-truth future weather.

**Track B: ERA5-pretrained → CMIP6 native transfer (additional, not a substitute).**
Compare data semantics, spatial grid and lead carefully; distinguish adaptation with limited CMIP6 historical training from zero-shot transfer. Never describe daily mean `tas` as an exact +24h instantaneous ERA5 target; use explicit harmonization/retraining, mandatory skill-drop controls and independent source-domain baselines.

## Critical checks BEFORE actual future training or regridding

1. Catalog search must return a matched historical and both SSP datasets with identical `source_id, member_id, grid_label`, daily `tas`. Distinct versions of same model member are not independent samples.
2. Verify each selected Zarr path exists, can be opened anonymously, includes `tas`, `time`, `lat`, `lon`, units in Kelvin and valid model calendar, with appropriate ensemble identity. Calendar `360_day` is not aligned by naïve Gregorian indexing.
3. Inspect historical date range (CMIP6 historical usually terminates in 2014) and future scenario start (typically 2015) from *actual arrays*, not catalog strings. Holdout period and historical fold selection must reflect the actual member coverage.
4. Model grid resolution and longitude wrap differ from ERA5 17×17. Require explicit conservative or bilinear spatial alignment policy and use latitude-area weighting only when appropriate on native grid. Never silently interpolate evaluation targets.
5. For day/tas, derive daily forecast inputs using daily-mean `tas(t)` and `tas(t+1 day)` **within each experiment and calendar**, then document the separate task; do not map daily `tasmax` or `tasmin` to mean `tas`.
6. Use strict time and split isolation; train/validation threshold reference computed exclusively on historical training years. Do not tune weights using future SSP trajectories.
7. Freeze model families and three loss candidates from historical work (tail-weight parameters 0, 1, 3), adjust only scientifically necessary new-domain training protocol with independent historical validation.
8. Report both `ssp245` and `ssp585` and, if sufficient coverage, at least two climate source models/realizations. Scenarios are *projections*, not observed future truth; inferences about robust future observations are invalid.
9. Report per-scenario high-tail precision/recall, false-positive rate, conditional >+2σ RMSE/bias, global RMSE and mean climate-change-signal fidelity. Score climate signal retention against **the driving model's projected change**, not observed ERA5 future truth.
10. Quantify uncertainty by independent climate-model member and meteorological time-block resampling; do not treat spatial grid cells as iid climate realizations. Record units, calendars, regridding operator, training dates, model ids, versions and hashes.

## Metadata source and primary traceability

- Pangeo CMIP6 Google Cloud Zarr catalog: `https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv`
- Pangeo documentation: https://pangeo-data.github.io/pangeo-cmip6-cloud/accessing_data.html
- NASA NEX-GDDP-CMIP6: https://www.nccs.nasa.gov/data-collections/nex-gddp-cmip6/
- NASA dataset is **daily** and downscaled; use it as a distinct impact-scale benchmark and beware bias correction. Do not mix those fields with the native CMIP6 catalogue without explicit provenance.
- Bounded metadata-only script: `scripts/audit_cmip6_catalog.py`; tests: `tests/test_cmip6_catalog.py`.
- One-off catalog audit workflow: `.github/workflows/e1-cmip6-catalog-audit.yml`. Until it succeeds, data pairs are candidate *search criteria*, not verified available stores.

## Go/no-go gate

**GO to small representative read only after** catalog audit produces actual model-member matched entries. **GO to domain-shift model training only after** time/calendar, variable/unit, grid, coverage and full training/evaluation split audits pass. This stage explicitly avoids training and downloading raw climate grids to reduce costs. Forecast changes and scientific significance remain unmeasured.
