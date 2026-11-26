# E1 v5.3 → v5.4 BCC cross-GCM replication, independent-source governance

## Verified prerequisites
- BCC source native regional 23×23 Kelvin daily `tas` and native noleap date pairs: [v5.0 verified](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868440215).
- 900 Historical training + 240 Historical validation regional source/target pairs, native noleap CF dates and 75 Historical training-only monthly P95 values per gridcell: [v5.1 verified](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868844059); frozen NPZ SHA `a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452`.
- Exact source-locked two-epoch BCC FNO/U-Net test, checkpoint save and reload: [v5.2 verified](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45870280521). Historical-only validation weighted RMSE K: Persistence 2.81204; FNO 2.74671; U-Net 3.10182. These are not fully trained nor future SSP results.

## v5.3 — full BCC training, as originally preregistered

[Three independent weight-arm CI jobs](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45870654954) were started *after* v5.2 passed.

- AWI-compatible architecture families (BCC fitted independently): FNO2d width8/modes4/depth2 and U-Net2d width8.
- Three seeds 11/22/33, three loss weights 0/1/3 (hot event multiplier 1×/2×/4×); 100 epochs, batch8, Adam LR 1e-3.
- Train-only BCC mean/SD and independently fit BCC monthly local P95, native BCC 23×23 grid.
- Validation checkpoint selection is by **unweighted BCC Historical 2005–2014 MSE**, irrespective of tail weight. Report **all** 18 model checkpoints and individual seed Historical metrics.
- SSP245/585 Zarr stores must not be accessed during training or checkpoint choice.
- Every checkpoint must contain immutable BCC dataset SHA and BCC model/local calibration, with its own checkpoint SHA. All three independent jobs must pass, no cherry-picked arm or seed.
- A poor two-epoch U-Net score does **not** justify dropping it from the prescribed full experiment.

**Status: running; no final v5.3 BCC 100-epoch metrics verified at document creation.**

## v5.4 — prepared *separate* BCC SSP evaluation, not triggered until v5.3 completes

`scripts/evaluate_cmip6_bcc_v54_frozen_ssp.py` and `.github/workflows/e1-cmip6-v54-bcc-frozen-ssp.yml` enforce:

1. Load the same immutable Historical v5.1 SHA and all **18** previously archived v5.3 checkpoint SHA values **before** opening any BCC SSP fields.
2. Independently recompute BCC native-calendar training-only 12×23×23 local gridcell P95, normalization and training-count constraints and compare to saved thresholds.
3. Read the exact predeclared SSP245 and SSP585 paired temperature schedule: **2030, 2050, 2080**, each month's **8th and 20th** native noleap day, giving 72 pairs per scenario. Do not silently cast to Gregorian dates, interpolate to AWI grid, or change study dates based on future results.
4. Score Persistence and all BCC Historical-frozen FNO/U-Net models for all scenario dates and each selected future year: area-weighted RMSE/MAE/bias; historical threshold local P95 Recall/Precision, true and predicted hot fractions; all three seed metrics and seed spread.
5. Save only a machine-readable JSON diagnostics/provenance archive (not full raw climate fields) with real experiment source paths, checkpoint hashes, and strict non-SSP-fitting assertions.

The BCC SSP data have previously been examined only in the v4.9/v5.0 *source feasibility* checks and small Persistence QA, not used for neural model fitting/checkpoint selection. Treat v5.4 as a frozen sample evaluation, **not a completely untouched source-data discovery process**.

## Planned cross-GCM scientific analysis after successful v5.4

Produce a matched **AWI vs BCC**, SSP245 vs SSP585, FNO vs U-Net, 1×/2×/4× grid of same-scenario Persistence-relative RMSE deltas and same-source local-P95 hot Recall deltas. The underlying grids, native calendars and absolute Kelvin distributions differ: compare *within-GCM* architecture/loss contrasts, and then their directions across two GCMs, not raw temperature-error averages pooled across climate-model gridcells.

Only two distinct CMIP6 GCMs and one realization/member each: no general CMIP6 ensemble significance, SSP-wide daily skill, global climate attribution or observed-future truth claim. Sparse samples do not support heatwave duration metrics.
