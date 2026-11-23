# E1 v0.8 — dataset structural audit

The WeatherBench 2 ERA5 Zarr path is documented by the official data guide:
https://weatherbench2.readthedocs.io/en/latest/data-guide.html

**What changed:** every newly exported NPZ contains `X`, `y`, `years`, and `target_times` (numpy datetime64 vector). The existing `load_npz_pairs` interface still returns X/y/years for compatibility, but validates timestamps when present.

```bash
python -m pip install -e '.[test,torch,era5]'
python -m pytest
python scripts/export_wb2_multiyear.py --years 2010 2015 2020 \
  --max-per-month 8 --output data/processed/e1_multiyear_dev.npz
python scripts/audit_e1.py --input data/processed/e1_multiyear_dev.npz \
  --metadata data/processed/e1_multiyear_dev.json \
  --output outputs/e1_v08_audit.json
```

The audit rejects duplicate, unsorted or inconsistent timestamp-years; mismatched checksums, sample counts, or metadata bounds; unexpected target variable or forecast lead. It reports year/month sample counts and provenance metadata. Legacy files without timestamps can still be loaded for old prototype tests, but fail the new provenance audit by default.

**Limitations:** The audit cannot verify that the source bucket is authentic, that each predictor occurred exactly 24h prior to the corresponding target, that ERA5 licensing conditions have been satisfied, or that region/split selection is scientifically defensible. Those require independent source and adapter verification. The selected three non-contiguous years and sparse monthly samples remain DEVELOPMENT ONLY.

**Critical gate:** This repository has not produced verified real ERA5 training results. Public documentation confirms the data source URL; this is not the same as confirming the dataset has been downloaded or the GitHub CI has passed.
