# E1 v0.9 — real WeatherBench 2 access gate

## Verified information

The official WeatherBench 2 data guide lists this ERA5 1.5-degree store:
`gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`

Official reference: https://weatherbench2.readthedocs.io/en/latest/data-guide.html

The guide says to inspect dataset-specific LICENSE files. Publicly readable does **not** mean unrestricted reuse.

## GitHub Actions

Workflow: `.github/workflows/wb2-access-probe.yml` (push or manual dispatch).

The probe:
1. Opens the GCS store **anonymously**, without user cloud credentials.
2. Reads metadata, variable names/dimensions, unit attribute and earliest/latest timestamps.
3. Reads a tiny 2 × 2 × 2 temperature subset and checks finite values.
4. Writes `outputs/wb2_access_probe.json`, even when the probe itself fails.
5. Uploads that JSON as the workflow artifact `wb2-access-probe`, retained for 14 days.

**Do not confuse the action being pushed with it passing.** Until an actual Actions run and diagnostic JSON are inspected, remote access remains unverified.

## Interpreting outcomes

- `status: success`: Cloud dataset can be accessed from that runner with the tested Python environment. This is a connection smoke test only.
- `status: failure`, `stage: gcsfs_anonymous_connect`: Cloud authentication or path setup issue.
- `stage: xarray_open_zarr`: Zarr metadata / dependency / location issue.
- `stage: metadata` or `coordinate_read`: Unexpected schema or time coordinates.
- `stage: tiny_chunk_read`: Actual chunk read, network or data issue.

## Scientific work still pending

Even a successful probe does not verify all 24-hour pairs, 2010/2015/2020 sampling, ERA5 license conditions, complete calendar handling, valid chronological OOD splits, model skill or future warming response. No climate-science result should be reported based on this probe.

The existing multi-year development exporter still produces sparse sampled months and loads the resulting subset into memory. A scientifically robust E1 dataset requires continuous year blocks, season/day-of-year climatology and independent test periods.
