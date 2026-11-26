# E1 v5.3 verified BCC Historical validation; v5.4 frozen SSP audit

## Frozen BCC Historical 100-epoch model comparison VERIFIED

[Successful v5.3 three-job GitHub workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45870654954). All 18 BCC FNO/U-Net checkpoints across three initialization seeds (11,22,33) and three training hot weights (1×,2×,4×) passed data SHA and output checks. Training exclusively BCC-CSM2-MR r1i1p1f1 gn historical 1980–2004 (900 sparse daily pairs, training-only normalization and 75/month P95) and checkpoint selection on Historical 2005–2014 validation (240 sparse pairs). Native `noleap` calendar; 23×23 regional grids. Source Historical NPZ SHA256 `a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452`.

### BCC Historical validation (mean of seeds, cosine(latitude) weighted RMSE in K)

| Architecture | Training hot multiplier | RMSE K | Recall vs BCC train monthly local P95 | Precision |
|---|---:|---:|---:|---:|
| Persistence | — | 2.812038 | — | — |
| FNO | 1× | **2.372459** | 0.493427 | 0.658298 |
| FNO | 2× | 2.377630 | 0.557726 | 0.621029 |
| FNO | 4× | 2.386163 | **0.573301** | 0.609591 |
| U-Net | 1× | **2.357095** | 0.497688 | 0.631677 |
| U-Net | 2× | 2.360268 | 0.489304 | **0.658011** |
| U-Net | 4× | 2.385764 | **0.501314** | 0.631293 |

**Within BCC Historical:** Both fully trained FNO and U-Net outperform native Persistence mean RMSE; U-Net 1× marginally outperforms FNO 1× (unlike AWI's FNO advantage). The strong hot weight raises FNO Recall from 0.4934 to 0.5733 while worsening its RMSE and reducing Precision. U-Net 2× Recall is below U-Net 1×; do not say all weighting monotonically improves detection. The validation set was used for checkpoint selection, so no inference that these are unbiased independent test scores.

Artifacts: [1× loss](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45870654954/artifacts/15142742746), [2×](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45870654954/artifacts/15142857935), [4×](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45870654954/artifacts/15141774430).

## v5.4 first BCC SSP245/585 **future** forecast: launched, results unverified

[Current BCC v5.4 test run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45872811683) was launched only after all three v5.3 training arms succeeded. Must verify **all 18 immutable checkpoints and source Historical SHA before any SSP grid is opened**. Predeclared BCC native day `tas(t)→tas(t+1 native day)` simulated SSP projected targets at 2030, 2050, 2080, two fixed days/month/year, 72 days per SSP245 and SSP585. All FNO/U-Net architecture and tail weight/seed arms plus Persistence are evaluated on each scenario-year.

No SSP model fitting, checkpoint selection, monthly threshold recalibration, training-data standardization, or source-grid remapping is allowed. BCC native `noleap` time, 23×23 grid and independent training-only thresholds must be used; AWI native `proleptic_gregorian` 26×27 is incompatible without regridding and calendar harmonization.

**Do not report v5.4 SSP scores as measured until its CI evaluation and archived JSON succeed.**

## v5.5 AWI–BCC cross-GCM comparison prepared

[v5.5 no-pooling comparison script](../scripts/compare_cmip6_v55_awi_bcc.py) and [guarded future workflow](../.github/workflows/e1-cmip6-v55-cross-gcm.yml) compare same-model Persistence-relative RMSE improvement and BCC/AWI own-Historical threshold hot recall differences. Must use *both* verified full v4.8 AWI and v5.4 BCC frozen SSP reports and preserve all 96 scenario × year-scope × GCM × architecture × loss-arm contrasts. Do not pool source grid cells across GCMs to fabricate uncertainty significance. A second climate model-member adds structural replication, not ensemble-complete evidence.

**At creation of this record, v5.5 has not been triggered.** The BCC v5.4 test must actually succeed before v5.5 source-matched contrast.
