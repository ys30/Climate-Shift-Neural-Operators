# E1 v5.19 — Full three-independent-GCM frozen SSP forecast comparison

## Data lineage and provenance
Reuse archived and independently successful GitHub Actions only: AWI v4.8 run 45864317174, BCC v5.4 run 45872811683 and CAMS v5.18 run 45905955958. All three trained **their own independent native Historical-only** FNO/U-Net models (18 checkpoints each; 54 across three GCMs). The source GCM SSP ensembles are simulated climate scenarios, not future measurements.

## Locked analysis
Three distinct source GCMs AWI-CM-1-1-MR, BCC-CSM2-MR and CAMS-CSM1-0; one model member each. Each member has two SSP scenarios (245 and 585) with 2030/2050/2080 sampled at two dates/month per year, 72 +1-day target pairs per GCM×SSP. 3 GCM × 2 scenarios × 2 neural architectures × 3 training hot weights = **36 arm-aggregated rows**, each neural arm has 3 historical-training seeds.

Compute area-weighted RMSE skill = 100 × (1 − neural RMSE / own GCM–SSP Persistence RMSE). Never pool Kelvin raw RMSE across native GCM regions/grids. Per GCM×SSP report standard-loss FNO skill, FNO 4×–1× hot Recall change, FNO 4×–1× weighted RMSE cost and Persistence–FNO4 hot Recall gap. Show all six cases and three clustered GCM outcomes transparently, including all counterexamples.

## Statistical caution
Two SSPs do **not** double independent-GCM replication count. Three distinct GCMs still do not justify inferential probability statements about all GCM climates. Fixed 72 sparse pairs/SSP cannot establish duration of continuous heatwaves, climate warming trends, physical conservation response or signal fidelity. The 2005–2014 Historical validation cohort was used to select epochs, not an untouched held-out test. Do not choose best neural training weights based on SSP results.

## Next research priority
Continuous daily multi-year evaluation with compute budgeting and additional GCM members/independent realizations. Then evaluate extreme-duration statistics and seasonality, with explicit serial-dependence uncertainty.
