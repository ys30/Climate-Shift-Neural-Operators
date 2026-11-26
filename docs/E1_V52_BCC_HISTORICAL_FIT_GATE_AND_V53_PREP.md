# E1 v5.1 success and v5.2–v5.3 independent BCC Historical fitting gates

## Verified BCC Historical-only source dataset (v5.1)

**Successful GitHub source extraction:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868844059  
**Exact archived BCC Historical-only NPZ and JSON:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868844059/artifacts/15142736094

- Source: **BCC-CSM2-MR r1i1p1f1 gn**, real CMIP6 daily mean `tas` in Kelvin, 25–50°N and 125–100°W, exact **23×23** native grid (529 cells).
- Correct native CF calendar: **noleap / 365_day** — `datetime64` Gregorian daily timestamps were **not** fabricated.
- Historical **1980–2004 train: 900** sampled daily source/target pairs (days 6,16,26/month/year); **2005–2014 validation: 240** pairs (days 8,20/month/year). Every sampled day has exact **+1 BCC native CF day** target and finite Kelvin region.
- Every training calendar month has **75** targets; independently fitted BCC-only **12×23×23** local gridcell P95, not imported AWI thresholds.
- Historical validation Persistence cosine(latitude)-weighted RMSE **2.8120375851 K**. This reflects BCC's own model and grid; it must not be interpreted as a causal effect of climate-model identity relative to AWI's 2.3298 K.
- **Immutable BCC Historical NPZ SHA256:** `a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452`.
- SSP245 and SSP585 **not read** by the v5.1 training-cohort extraction.

## BCC v5.2 true Historical FNO/U-Net smoke

[One-time v5.2 actual Historical-only FNO/U-Net training](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45870280521) was launched after verification of v5.1. Config: frozen source SHA, seed11, **two epochs**, batch8, FNO width8/modes4/depth2, U-Net width8, Adam 0.001, standard MSE, model checkpoint selected by unweighted BCC Historical validation MSE. Its code is `scripts/fit_cmip6_bcc_v52_historical.py`, with source/no-leap and metric safety tests. Each saved checkpoint includes own BCC train-only normalization and local-month P95 and must reload with `torch.load(...,weights_only=True)`.

**Status is pending until CI/model artifact success.** Two-epoch results may underperform Persistence and are primarily a training feasibility check.

## BCC v5.3 frozen full matrix, prepared but NOT launched

A separate `scripts/run_cmip6_bcc_v53_historical_arms.py` implements full fixed **2 architectures × 3 seeds (11/22/33) × 3 tail-weight arms (0,1,3) = 18 model checkpoints**, 100 epochs per model, Historical train/validation only, and source SHA safety checks. Full GitHub Actions workflow is available at `.github/workflows/e1-cmip6-v53-bcc-historical-arms.yml`, plus `tests/test_cmip6_bcc_v53_historical_arms.py`.

It must **not be triggered until v5.2 Historical fitting and checkpoint reload have succeeded**. These arms will all be reported, with area-weighted RMSE/MAE/bias, local P95 Recall/Precision, predicted hot fraction and seed dispersion. Historical validation is for checkpoint selection and not an independent Historical test. Candidate weight arms are frozen in advance and are **not tuned on the previous AWI v4.8 SSP projections**.

## Following future-domain test and source-model uncertainty

Only after BCC has produced independently fitted/frozen checkpoints should SSP245 and SSP585 (2030,2050,2080, days8/20 per month) be evaluated on BCC native **noleap** 23×23 grids. Both projected scenarios require no target-based refit, all model arms, identical BCC source/target +1 native day and independent BCC Historical P95. All future results are source-GCM simulated projected conditions, **not observed future weather**.

Even if AWI/BCC effect directions agree, this is just **two CMIP6 structural models with one member each**, plus three weight seeds. No claim of full ensemble reproducibility, unbiased SSP-wide continuous sampling or observed warming signal retention.
