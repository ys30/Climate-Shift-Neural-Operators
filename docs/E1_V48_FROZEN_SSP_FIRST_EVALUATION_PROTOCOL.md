# E1 v4.7 VERIFIED; v4.8 first frozen Historical-to-future SSP forecast experiment

## Verified v4.7 source and frozen validation results

- [Complete v4.7 3×3 Historical-only model training experiment](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45861178303): all **three jobs passed**, and **all 18 checkpoints** (FNO/U-Net × 3 seeds × 3 loss weights) passed per-file SHA256 validation.
- [Standard MSE checkpoint artifact](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45861178303/artifacts/15138743116)
- [2× hot-weight checkpoint artifact](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45861178303/artifacts/15139245582)
- [4× hot-weight checkpoint artifact](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45861178303/artifacts/15139335267)
- Historical training: native AWI-CM-1-1-MR day/tas 1980–2004, 900 sampled daily +1-day pairs, 26×27 grids, 75 samples per training calendar month for local P95; Historical validation: 2005–2014, 240 pairs; all normalization/P95 fit **training-only**. Dataset SHA256 `ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914`.
- 100 epochs; seeds **11,22,33**; hot-temperature training multipliers 1×, 2×, 4× (tail_weight 0,1,3).

### Actual Historical validation — mean across 3 training seeds

| Model | Tail training multiplier | Cos(lat) RMSE K | Local monthly P95 Recall | Local monthly P95 Precision |
|---|---:|---:|---:|---:|
| Persistence | — | 2.329778 | not aggregated here | not aggregated here |
| FNO | 1× | **2.034273** | 0.551224 | 0.664075 |
| FNO | 2× | 2.039330 | 0.553817 | **0.666928** |
| FNO | 4× | 2.050005 | **0.591545** | 0.653723 |
| U-Net | 1× | **2.048560** | 0.528598 | 0.647864 |
| U-Net | 2× | 2.048727 | 0.514857 | **0.665873** |
| U-Net | 4× | 2.053993 | **0.553126** | 0.640702 |

FNO 1× reduces Historical validation RMSE versus Persistence by about 12.7%. Strong 4× tail weight increases monthly P95 Recall but worsens mean RMSE and, for both architectures, slightly lowers mean precision. The 2× arm does **not** replicate the ERA5-era small improvements simultaneously across metrics in a stable way. Seed-level variability and checkpoint selection on this same Historical validation set mean it is not a prospective independent test. No significance claims from a single AWI climate model-member.

## Frozen v4.8: future-projection evaluation, not model selection

[One-time future-domain workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174) was triggered **after all v4.7 arms were verified**.

- Before opening any SSP data, it downloads the immutable Historical dataset and all 18 checkpoints, checks source artifact SHA and per-checkpoint hashes, model family, seed, tail multiplier, normalization, Historical training-only local P95, and allowed epoch/architecture.
- It evaluates **ALL** 18 frozen model checkpoints, not a winner chosen after reading SSP.
- The two scenarios are **SSP245 and SSP585**, AWI-CM-1-1-MR `r1i1p1f1 gn`, native Western-US 26×27 daily `tas`. The exact future source-target schedule was fixed in v4.4: **2030, 2050, 2080; days 8 and 20 in each month**. 72 sparse pairs per SSP; 24 pairs per sampled year.
- Pretrained FNO/U-Net predictions and Persistence are scored on each scenario in aggregate and for each of the three years. Metrics: area-weighted RMSE/MAE/Bias, Historical-trained local monthly P95 Recall/Precision, predicted hot-event fraction, per-seed means and spread. SSP data may not be used for gradient updates, threshold fitting or checkpoint selection.
- Reports will be archived as **derived JSON with checkpoint provenance**, no raw SSP temperatures or checkpoint modifications.

### Key study limitations

This is *the first actual planned prospective SSP domain-shift evaluation of an AWI simulated future climate*, but still **not independently observed future weather truth**, a multi-model climate ensemble, or complete continuous-day sampling. Earlier v4.4 SSP temperature pairs were read for data-quality/Persistence checks, so these years are not wholly unseen by the research process; however, **no SSP target was used in model fitting, training climatology, loss setting, validation checkpoint choice or parameter tuning**. Treat the future model predictions strictly as a sparse fixed-domain evaluation. Do not infer heatwave durations from the two dates/month design. Do not conflate CMIP6 daily-mean targets with ERA5 instantaneous +24h temperature fields.

**At document creation, v4.8 future forecast results remain pending; avoid claiming success without inspecting its Actions run and artifact.**
