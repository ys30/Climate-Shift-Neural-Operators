# E1 v5.12 — independent 2040 sampled-date evaluation of frozen trained checkpoints

## Gate and purpose
The v5.11 real source-only native CMIP6 2040 date pilot was independently verified in GitHub Actions run 45893094787. Apply *all* historical-only AWI v4.7 and BCC v5.3 trained checkpoints, with fixed normalizations, local monthly P95 thresholds and original architectures, to new target dates. No gradient descent, no checkpoint reselection and no changing hyperparameters based on projected temperatures.

## New future cohort
Year 2040; source days 3,7,11,15,19,23,27 per month (84 pairs/scenario). Exact native +1 day target pairing. Two independent native GCM source grids (AWI 26×27 and BCC 23×23), each tested separately on SSP245 and SSP585. This is a supplementary test introduced after examination of earlier projections; not pristine model-development blind evaluation. Re-read native 2040 fields and verify regional source and target temperature against the original v5.11 archived results to guard data drift.

## Model factorial
2 GCM × 2 SSP × (Persistence + 2 neural architectures × 3 hot weights × 3 training seeds). All **36 distinct frozen checkpoints** are evaluated across their own two SSP scenarios. Store per-seed area-weighted RMSE/MAE/bias and training-defined monthly local P95 Recall/Precision. Preserve FNO 1×/2×/4× versus own persistence, and report tradeoffs without selecting a winner based on SSP scores.

## Publication limitations
The 84 sampled dates per scenario in one year are not a continuous climate annual mean, multi-year forced warming trend or heatwave duration test. GCM SSP outputs are simulation targets, not independently observed future weather. Two GCMs, one member each, cannot establish ensemble-wide inference. No pixel-based hypothesis tests; seed variation does not represent climate internal variability or GCM diversity.

## Next scientific priority
Dense multi-year evaluation with appropriately sized computational budget, new GCM/member replication and continuous daily data for extreme-duration metrics.
