# E1 v4.6 VERIFIED Historical smoke; v4.7 frozen historical three-arm experiment

## v4.6 successfully verified

- [End-to-end successful 2-epoch model-fit run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45859545722)
- [Frozen Historical-only FNO and U-Net checkpoints and diagnostics](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45859545722/artifacts/15137711474)
- Exact immutable v4.5 AWI-CMIP6 Historical dataset SHA256: `ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914`. Verified 900 Historical training (1980–2004) and 240 Historical validation (2005–2014), every month train count = 75 and local monthly gridcell P95 fitted training-only. No SSP input was read.
- Two-epoch, single-initialization seed 11, standard unweighted MSE, fixed width8, FNO modes4/depth2. Frozen checkpoint and metadata hashes all verified.

| Predictor | Historical 2005–2014 cos(latitude)-weighted validation RMSE K |
|---|---:|
| Persistence, no training | **2.32977785** |
| FNO, 2 epochs seed 11 | 2.48196660 |
| U-Net, 2 epochs seed 11 | 2.72976931 |

The two neural models underperformed Persistence after just two epochs. This is a functional smoke, **not** proof of a model architecture defect, a fully converged comparison, a future climate-shift result, or a generalization breakthrough. No hyperparameter has been selected from SSP.

## v4.7 predeclared and triggered

**[One-time 3-weight × 3-seed × 100-epoch workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45861178303)**

| Factor | Frozen value |
|---|---|
| Models | FNO2d, U-Net2d; Persistence as fixed validation baseline |
| Training source | AWI-CM-1-1-MR native Historical daily mean tas, 1980–2004 |
| Validation source | Same climate-model/member/grid Historical only, 2005–2014 |
| Forecast task | Same-source daily tas(t)→tas(t+1 native day), 26×27 Western US |
| Initialization seeds | 11, 22, 33 |
| Training loss treatments | standard MSE weight 0 (1× hot sample), moderate weight 1 (2×), strong weight 3 (4×) |
| Epochs | 100 per seed/treatment/model; validation checkpoint selected by *unweighted* Historical MSE |
| Batch / optimizer / hyperparameters | Batch 8, Adam 0.001, width8, FNO modes4/depth2, U-Net width8 |
| Normalization and local P95 cutoffs | Fit using 900 Historical training samples only; 75 source-month targets each |
| Expected saved checkpoints | 18 total (2 model families × 3 weights × 3 seeds), all with source-data SHA and checkpoint SHA |
| Job structure | Three parallel or queued weight-specific jobs (each three seeds), timeout 90 min per job, no retrigger on code changes |

Success requires **all three arm jobs**, full 3-seed validation metrics (weighted RMSE, MAE, bias; local P95 Recall and Precision; predicted hot fraction), metadata and checkpoint SHA verification. All candidates must be reported, not only the best. Dataset derived from one native climate-model member with sparse daily sampling; no SSP data is read.

## Planned next scientific gate, not yet executed

1. Freeze each trained checkpoint and normalization/P95 on Historical-only.
2. Create an independent, read-only SSP245 and SSP585 loader that matches the exact same grid and +1-day target, using **predeclared** scenario years and dates; no new weights or training.
3. Produce same-scenario Persistence and each frozen FNO/U-Net weighted error and local hot fraction (using Historical training thresholds), show per-scenario/per-year differences and any warming-signal bias.
4. Correct for model/member structural differences, carefully separate shift from source climate-model simulated warming, and explicitly limit claims for sparse samples and one member.

**No v4.7 trained-model validation results have yet been verified, and no future SSP forecast skill exists in the repo as of this document.**
