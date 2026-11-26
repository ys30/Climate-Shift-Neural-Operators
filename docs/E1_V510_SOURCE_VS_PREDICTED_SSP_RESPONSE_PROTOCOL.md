# E1 v5.10 — Native physical SSP contrast and frozen prediction response

## Purpose
After v5.8 (frozen SSP585–SSP245 forecast error) and v5.9 (native simulated source temperature reference), join *independently archived, unchanged* metrics for a clearer descriptive science table.

Within one GCM and one projected year, with identical dates and area weights:
- Source `S = mean(target_SSP585) - mean(target_SSP245)`.
- Model error `E = bias_SSP585 - bias_SSP245` from previously frozen v5.8.
- Reconstructed model SSP response `P = S + E`.

This is an **algebraic reconstruction of sampled regional-mean contrasts**, not new predictions nor an estimate of forced warming. No SSP checkpoint, loss weight or model hyperparameter is re-selected.

## Factorial and safeguards
2 GCMs × 3 sampled future years × 7 predictor arms (Persistence, FNO 1×/2×/4×, U-Net 1×/2×/4×) = 42 joined source/predictor records. Require v5.8 and v5.9 exact statuses, sample schedules, all rows, finite Kelvin means, duplicate absence, consistent historical references, physical range and per-seed consistency.

AWI native proleptic Gregorian 26×27 and BCC native noleap 23×23 remain independently averaged. Report signed error and absolute error, not pooled pixel significance.

## Explicit limitations
- One member each from only two GCMs; model simulations are not observed future truth.
- Target-date sampling 24 days/year can mix internal variability with climate forcing.
- Historical reference 2005–2014 was used for historical checkpoint selection; it is not an untouched holdout.
- Joining regional means does not provide spatial pattern preservation, future heatwave duration, or full climate-state sensitivity.
- Never select weight arms or claim ensemble-wide significance on these differences.

## Next gate
Build a *separately versioned, predeclared* dense daily multi-year CMIP6 sample from existing native Zarr, then apply all fixed historical-trained checkpoints, with proper computational/scientific safeguards. Preserve early sparse tests as exploratory prior evidence, not as pristine blind future tests.
