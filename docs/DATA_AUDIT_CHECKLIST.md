# E1/E2 data-access and provenance checklist (not yet verified)

Do not commence full training until this card is filled with source URLs and hashes/versions.

## E1: ERA5 / WeatherBench 2

- [ ] Exact provider, download URI, access date, license/terms, citation
- [ ] Grid, spatial mask, resolution, coordinate convention, and land/sea treatment
- [ ] Variables, pressure levels, units, temporal sampling and prediction lead
- [ ] Chronological split with embargo, validation-only selection, held-out test
- [ ] Training-only normalization; train-derived heat thresholds and reference climatology
- [ ] Baselines matched on input information and tuning budget

## E2: CMIP6

- [ ] Institution/model, experiment, member, version, calendar, grid labels
- [ ] historical/SSP2-4.5/SSP5-8.5 paired member availability
- [ ] Historical baseline and future windows explicitly fixed
- [ ] Distinguish within-model and leave-one-model-out generalization
- [ ] Regridding diagnostics and conservation/area-weighting choices
- [ ] Distinguish emulator output from observed ground truth

## Reproduction manifest

Record commit, config, seeds, preprocessing version, checksum, and compute environment. No raw ERA5/CMIP6 data committed to git by default.
