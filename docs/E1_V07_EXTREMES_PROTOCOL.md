# E1 v0.7 — preliminary heat-tail and multi-seed diagnostics

## Implemented

- Fit a fixed 95th percentile threshold from **training target values only**; validation and test do not influence the threshold.
- Score FNO, U-Net and persistence with the **same** threshold on held-out samples.
- Report reference and predicted grid-cell exceedance fractions, frequency bias, conditional temperature bias, precision and recall.
- Optional cosine-latitude weights are available in `hot_diagnostics`, but the current `compare_e1` pipeline **does not yet carry latitude coordinates** and therefore its reported tail metrics are unweighted.
- Undefined no-event metrics are serialized as JSON `null`, not 0 or 1.
- Run independent seeds with per-seed files plus descriptive means and standard deviations. Across-seed SD is **not** a confidence interval and cannot replace sampling uncertainty.

## Run

```bash
python -m pip install -e '.[test,torch,era5]'
python -m pytest
python scripts/run_e1_multiseed.py \
  --input data/processed/e1_multiyear_dev.npz \
  --output-dir outputs/e1_v07_multiseed \
  --train-end 2010 --val-end 2015 --test-end 2020 \
  --seeds 11 22 33 --epochs 4 --width 8 --modes 4
```

The requested years are example development years; do not cite these sparse data as a climate-shift experiment.

## Scientific limitations / required gates

1. A **global threshold across all seasons and grid cells is not a climate-standard heatwave definition**. Replace with training-period day-of-year/region-specific climatology and multi-day persistence before any event-fidelity claim.
2. Input data are still not a verified multi-year ERA5 extraction; remote Zarr access and CI results need independent confirmation.
3. Multi-seed variance is an optimization diagnostic, not uncertainty about climate-change response. Require time-block/bootstrap and model ensemble analysis.
4. Need latitude coordinate in extraction manifest, then pass it into area-weighted metrics; do not describe current results as area weighted.
5. Need spatial holdout design, weather predictor parity, model tuning parity, and CMIP6 future-minus-historical contrasts before paper conclusions.
