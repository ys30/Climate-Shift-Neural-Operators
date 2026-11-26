# E1 v4.8 — VERIFIED first frozen Historical-to-SSP CMIP6 daily temperature forecast experiment

## Outcome and exact provenance

- **Successful full workflow**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174
- **Unedited machine-readable full JSON**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174/artifacts/15140346151
- Historical v4.5 AWI-CM-1-1-MR `r1i1p1f1 gn` data SHA256: `ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914`.
- All **18 frozen** models were recovered from [v4.7 fully successful Historical-only training](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45861178303), with all checkpoint hashes and original Historical training-only local P95 thresholds verified *before opening SSP data*. The v4.8 workflow had **zero SSP training / normalization / threshold fitting / checkpoint reselection**.
- Models: FNO2d and U-Net2d; weights 0, 1, 3 correspond to warm training sample multipliers **1×, 2×, 4×**; three independent training initializations **11,22,33**, fixed 100 epochs, checkpoint selection using Historical 2005–2014 validation only.
- Test AWI model projected daily mean `tas` `t→t+1 native day`, Western-US native 26×27 grid. **SSP245** and **SSP585**, each 72 forecast pairs: exactly 2030, 2050 and 2080, 8th and 20th of all 12 months. Native local Kelvin values, cosine-latitude area weighting, and Historical-training-only local monthly P95.

## Table 1 — actual fixed SSP test RMSE, three seed mean (K)

| Architecture | Hot training multiplier | SSP245 area-weighted RMSE | SSP585 area-weighted RMSE |
|---|---:|---:|---:|
| Persistence | — | 2.171081 | 2.118762 |
| FNO | 1× | **1.9323** | **1.8836** |
| FNO | 2× | 1.9409 | 1.8889 |
| FNO | 4× | 1.9599 | 1.9049 |
| U-Net | 1× | **1.9552** | 2.0180 |
| U-Net | 2× | 1.9625 | **2.0087** |
| U-Net | 4× | 1.9635 | 2.0187 |

Relative to same-scenario Persistence, FNO 1× reduces RMSE by approximately **11.0%** (SSP245) and **11.1%** (SSP585). These gains are conditional on the **fixed sparse AWI simulated SSP test**, not a guarantee for future observations or other model members. The 2× training weight does **not** dominate 1× on overall RMSE in either SSP.

## Table 2 — local monthly historical-train P95 exceedance Recall

| Architecture | Hot multiplier | SSP245 Recall | SSP585 Recall |
|---|---:|---:|---:|
| Persistence | — | 0.7286 | 0.7962 |
| FNO | 1× | 0.679 | 0.763 |
| FNO | 2× | 0.681 | 0.764 |
| FNO | 4× | **0.708** | **0.786** |
| U-Net | 1× | 0.611 | 0.628 |
| U-Net | 2× | 0.624 | 0.641 |
| U-Net | 4× | **0.651** | **0.663** |

Strong hot training weight improves **mean** Recall across both architectures/scenarios, at the cost of worse mean RMSE. **Persistence has the highest aggregate Recall** in both scenarios despite higher regression RMSE: error scores and exceedance classification must be reported separately. Precision also changes and must not be omitted: for example, SSP245 FNO Precision 1× **0.743**, 2× **0.747**, 4× **0.735**; SSP585 FNO ~**0.775**, **0.775**, **0.769**.

## Per-year change under fixed Historical-trained hot threshold

These are fractions of sampled **grid-cell × date** values, NOT continuous heatwave days.

| Scenario | 2030 reference hot fraction | 2050 | 2080 | 2080 FNO 1× predicted hot fraction | 2080 U-Net 1× predicted hot fraction |
|---|---:|---:|---:|---:|---:|
| SSP245 | 0.1198 | 0.1950 | **0.3241** | 0.297 | 0.238 |
| SSP585 | 0.1489 | 0.2054 | **0.4918** | 0.484 | 0.385 |

This is a model-internal simulated climate shift relative to a fixed **1980–2004 Historical training threshold**. It is evidence of an increasing Historical-relative hot exceedance fraction within these sparse AWI scenario samples, not observational validation of real future warming. The FNO 1× hot fraction stays relatively close to the 2080 scenario reference; U-Net 1× is more conservative, underpredicting the simulated hot exceedance frequency. This finding requires independent model-member and calendar-sampling sensitivity analysis.

## Three-year total-RMSE disaggregation, standard-MSE models (K)

| Scenario | Year | Persistence | FNO 1× | U-Net 1× |
|---|---:|---:|---:|---:|
| SSP245 | 2030 | 2.0004 | 1.8345 | 1.8872 |
| SSP245 | 2050 | 2.4497 | 2.2249 | 2.1688 |
| SSP245 | 2080 | 2.0342 | 1.6988 | 1.7886 |
| SSP585 | 2030 | 1.8707 | 1.6468 | 1.6865 |
| SSP585 | 2050 | 2.1363 | 2.0157 | 2.0265 |
| SSP585 | 2080 | 2.3246 | 1.9667 | 2.2948 |

Annual ranking and magnitudes vary; never infer a monotonic forecast error or easy/hard SSP relationship from just three selected simulated years. FNO 1× outperforms Persistence in all six selected scenario-year cells, whereas U-Net 1× sometimes approaches Persistence (SSP585 2080).

## Interpretation and publication constraints

1. **This is a real model-domain generalization experiment** in the narrow sense of scoring Historical-trained frozen neural predictors on future **CMIP6 simulated** weather targets, NOT validation against independently observed future temperatures.
2. The native CMIP6 daily-mean `tas` target differs from the existing ERA5 instantaneous +24h target; do not merge absolute scores across the tasks.
3. The v4.4 data feasibility pilot had already sampled the same SSP dates and Persistence; no SSP data were used in neural training, parameter selection, thresholds or v4.7 candidate selection, but these test dates were **not entirely undisclosed to the research process**.
4. Only **one AWI climate-model member**, three nonconsecutive future years, two predeclared dates/month/year, and three initialization seeds. Grid cells and adjacent dates are not independent climate ensemble replicates. Do not claim statistical significance, global climate representativeness, full-century skill or heatwave episode duration.
5. The 1980–2004 Historical train has 75 samples/month across 25 years; approximately four >P95 tail observations per month per gridcell in expectation, so extreme-threshold estimates are exploratory.
6. Report SSP245 and SSP585, all 18 model checkpoints and Persistence on the exact same grid, units and dates. No hyperparameter changes are allowed post hoc based on these SSP results if claiming independent testing.
7. The key paper direction is **tail-awareness and model architecture affect mean error, high-temperature detection and simulated distribution-shift calibration differently**. Confirm with more CMIP6 models/members and richer heldout daily coverage before strong claims about climate-change generalization or retained warming signal.

## Immediate next reproducibility priorities

- Audit empirical paired seed deltas and annual-year variation without pseudo-replication. Plot mean RMSE vs Recall vs Precision and simulated hot-frequency bias by scenario, architecture and loss weight.
- Add independent second CMIP6 model/member with **actual** 2015–2100 SSP245+SSP585 stores; avoid the known ACCESS-CM2 SSP585 2201–2300 extension trap.
- Broaden sampling to more dates/month only under a **new, explicitly named follow-up** benchmark; retain v4.8 as frozen and avoid reselecting on it.
- Physically investigate calendar alignment, model resolution and whether projected warming signal is retained in model predictions; current reference hot fraction is not by itself a direct measured warming-signal amplitude.

**No claim of publication-ready robustness or full continuous-day/observational validation.**
