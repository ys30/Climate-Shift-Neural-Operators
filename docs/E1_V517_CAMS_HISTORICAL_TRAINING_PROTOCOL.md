# E1 v5.17 — Third independent CAMS native Historical FNO / U-Net training

## Strict prerequisite and source separation
The v5.16 CAMS Historical-only data (source 1980–2014) must have a successful audited GitHub run and an exact SHA256 sidecar, matching train/validation pair counts 900/240 and native grid/calendar. The v5.17 runner will fail closed if v5.16 has not succeeded. Do NOT load any SSP data before the 18 resulting checkpoints are frozen.

## Predeclared training
Fit native CAMS models (not transferred AWI/BCC checkpoints). Architectures: FNO2d with width 8, 4 Fourier modes per dimension, depth 2; UNet2d width 8. Optimizer Adam lr=0.001, 100 epochs, shuffled mini-batches size 8, fixed seeds [11,22,33], three training tail event weights [0,1,3] corresponding to 1×, 2×, 4× hot-loss multipliers. Choose each seed's epoch using Historical 2005–2014 **standard MSE** only. No SSP scoring, Hyperparameter tuning or checkpoint selection.

Calculate CAMS train mean and SD independently from 1980–2004 target fields, and CAMS native monthly local P95 from exactly 75 Historical training targets/month. Archive 2 architectures × 3 weights × 3 seeds = **18** immutable checkpoints in three GitHub artifact archives, with individual SHA, metrics and provenance.

## Evaluation status
Historical validation RMSE and P95 recall are *model-selection validation* results and should not be described as independent held-out verification. Future SSP245/SSP585 will only be evaluated in a separate later workflow (v5.18), after all 18 checkpoints are verified. The source data are sparse, not continuous heatwaves; only three GCMs with one member each will be available even after future evaluation.

## Execution
The workflow references v5.16 success run 45900398047 and requires its NPZ/JSON artifacts. Matrix jobs use 0/1/3 weights and each runs three seeds × two models for 100 epochs; no stochastic SSP-based selection.
