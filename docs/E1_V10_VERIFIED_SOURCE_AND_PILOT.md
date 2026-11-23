# E1 v1.0 — verified source and bounded real-data pilot

## Grounded source verification

GitHub Actions run [45103779035](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45103779035) completed successfully and its probe logs verify:

- Public `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr` read anonymously.
- `2m_temperature` present, unit `K`, dimensions time/longitude/latitude (the adapter accepts and transposes these).
- Actual first and last time coordinates: `1959-01-01T00:00:00`, `2023-01-10T18:00:00`. **Do not treat the publication-year label as implying complete 2023.**
- A real 2 × 2 × 2 temperature sample was read successfully, with observed sample range 284.03784 to 285.63794 K.
- Probe artifact: [wb2-access-probe](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45103779035/artifacts/14786821178).

Official source list: https://weatherbench2.readthedocs.io/en/latest/data-guide.html. Consult per-dataset license in GCS before redistribution.

## Limited real-data experiment

[Real-data pilot workflow](../.github/workflows/e1-realdata-pilot.yml) extracts 2010, 2015, 2020 (2 source times per month), validates NPZ vs JSON metadata and scores the same single-channel 24-hour temperature target for persistence, FNO, and U-Net. All results are archived in Actions artifacts, without committing ERA5 sample data to Git.

This workflow is a **smoke-and-integration test**, NOT a scientific train/test benchmark. Three isolated years, 2 samples/month and 2 model epochs do not support OOD or causal climate-change conclusions. Required before a paper: continuous-year windows; non-overlap/embargo; robust seasonality; geographic holdouts; compute-budget parity; physical constraints; multiple independent seeds; uncertainty, area weighting and CMIP6 evaluation.

## Next evidence gate

Inspect the actual pilot workflow conclusion and the archived source metadata / audit / metrics. A successful run proves extraction and data-to-model integration, not forecast skill sufficient for publication.
