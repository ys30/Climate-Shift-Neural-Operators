# E1 v2.5 — verified real ERA5 100-epoch three-seed convergence

**Successful run:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45556189440  
**Artifacts:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45556189440/artifacts/14996326295  
**Run commit:** `de77ec7e30e5b42a3e4341ced85f621b9a25dd82`.  
**Status:** SUCCESS — all unit tests, parameter audit, WB2 ERA5 extraction, chronological split and exact +24h target provenance, 100-epoch histories and output completeness checks passed.

## Experimental configuration

The same 1,440 sampled WeatherBench 2 ERA5 2-m-temperature pairs on a 17 × 17 grid at 25–50°N, 125–100°W. Train: 2008–2010 (864), validation: 2011 (288), test: 2012 (288). Each monthly cohort has 24 +24h pairs; not uninterrupted daily data. Fixed FNO width 8, modes 4×4, depth 2; fixed U-Net width 8. Optimization: Adam, 0.001, batch 8, 100 epochs, early best checkpoint chosen on validation. Random seeds 11, 22, 33.

## Observed results (mean ± across-seed SD; K unless noted)

| Model | Area-weighted RMSE | Area-weighted MAE | Global train-P95 recall | Monthly local train-P95 recall |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 3.461329 | 2.318198 | 0.738879 | 0.542995 |
| FNO | 2.847327 ± 0.005092 | 2.007119 ± 0.001083 | 0.710325 ± 0.015466 | 0.442778 ± 0.002403 |
| U-Net | 2.809589 ± 0.043752 | 1.986758 ± 0.026125 | 0.658676 ± 0.027588 | 0.419875 ± 0.021141 |

Individual weighted RMSE:
- FNO: 2.851019, 2.849445, 2.841518 K.
- U-Net: 2.857513, 2.771780, 2.799473 K.

**Critical nuance:** the seed-11 FNO is slightly better than U-Net at 100 epochs, while seeds 22 and 33 favor U-Net. **Do not describe the architecture ordering as stable.** Mean advantage of U-Net over FNO is about 0.038 K on this test year; this is not an architecture-level inference.

Compared with Persistence, mean area-weighted RMSE improves ~17.7% (FNO) and ~18.8% (U-Net), whereas monthly local-P95 recall is lower than Persistence by ~0.100 and ~0.123 respectively.

## Compare with earlier real run

- 5-epoch three-seed benchmark: run https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45478166832, weighted RMSE FNO 3.175610, U-Net 3.126368 K.
- 50-epoch **single-seed** run: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45554909392, weighted RMSE FNO 2.877988, U-Net 2.857513 K.
- 100-epoch **three-seed** mean: FNO 2.847327, U-Net 2.809589 K.

The increase from 5 to 100 epochs significantly improves mean skill on the same dataset, but the local extreme recall deficit persists. For convergence, inspect per-seed best epoch and validation curves inside artifact; **across-seed variance is not a climate-year uncertainty estimate**.

## Scope limits / science gate

This is short-horizon historical out-of-year forecasting, not the climate-shift distributional generalization study promised by the overall project title. Five calendar years with 24 samples/month and one 2012 test year cannot identify actual global-warming OOD skill or quantify heatwave duration. A three-seed sample is a reproducibility control, not an independent climate ensemble.

The next [matched-capacity pilot](../.github/workflows/e1-capacity-pilot.yml) is designed to compare FNO and U-Net at close **real scalar degrees of freedom**. Its 50-epoch, single-seed results will only be screening evidence; longer independent-year evaluation and future-climate shift still must follow.
