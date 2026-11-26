# E1 v4.1 — Verified CMIP6 catalog pairs; actual Zarr store audit in progress

## v4.0 evidence

- Successful one-time GitHub Actions audit: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45758551504
- Full compact metadata manifest: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45758551504/artifacts/15088103741
- Public original Pangeo catalog: https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv
- SHA256 of catalog at run: `e319cd2bf1daf9b5aa531f92c022d5322ee6bce0b566ac81dfae31dbae203fd9`.
- **238** matching `source_id/member_id/grid_label` candidate trios, representing **36** distinct CMIP6 models, each indexed for `day/tas` in `historical`, `ssp245`, `ssp585`.
- First eight selected entries from the actual catalog: ACCESS-CM2 r1i1p1f1 gn; AWI-CM-1-1-MR r1i1p1f1 gn; BCC-CSM2-MR r1i1p1f1 gn; CAMS-CSM1-0 r2i1p1f1 gn; CESM2 r10i1p1f1 gn; CESM2-WACCM r1i1p1f1 gn; CMCC-CM2-SR5 r1i1p1f1 gn; CMCC-ESM2 r1i1p1f1 gn.
- The earlier 35 MB read guard caused the original catalog workflow to fail; a bounded 160 MB guard succeeded. That prior failure was **not** evidence of missing climate data.

**What the v4.0 audit verifies:** public *catalog rows* with exact native ensemble, grid and experiment matches, no invented stores; valid catalog schema and metadata-only artifact; NO actual stores yet opened, no calendar or climate-array verification, and NO future model forecasting.

## v4.1 Zarr access gate

A separate one-time [read-only actual Zarr probe](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759128555) checks the first two model trios (6 stores) from [frozen, catalog-grounded candidate metadata](../experiments/cmip6_v40_first_two_catalog_triples.json):

- ACCESS-CM2, `r1i1p1f1`, `gn` — Historical / SSP245 / SSP585.
- AWI-CM-1-1-MR, `r1i1p1f1`, `gn` — Historical / SSP245 / SSP585.

Checks: whether metadata can actually be opened anonymously; daily `tas` has 3 dimensions, variable units are Kelvin, `time` is valid native CF calendar metadata, latitude and longitude coordinate arrays exist, native grid shape and member across arms match. The probe reads no full temperature fields, only Zarr metadata and two scalar time-coordinate endpoints. Results are **pending** until GitHub run success and audit JSON examination.

## Research gate and limits

Native CMIP6 `day/tas` is daily mean temperature and does **not** equal instantaneously sampled ERA5 +24h T2m. Existing ERA5-FNO/U-Net weights should not be presented as directly transferable to daily `tas` or NASA daily downscaled `tasmax`. Correctly define a **new native-CMIP6 daily +1 day** task or an explicitly harmonized joint-domain target. Keep training climatology and P95 thresholds historical-only, use exact native calendar and record SSP years. SSP projections do not provide observed future ground truth.

**No future CMIP6 skill, warming signal retention, daily extreme validation, or OOD claims yet.** The next gate must prove real Zarr access and calendar/time semantics before any expensive training. Read [v4.0 preregistered protocol](E1_V40_CMIP6_HISTORICAL_SSP_PROTOCOL.md).
