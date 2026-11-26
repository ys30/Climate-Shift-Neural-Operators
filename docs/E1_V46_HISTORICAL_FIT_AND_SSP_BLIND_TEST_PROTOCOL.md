# E1 v4.6 — No-leakage AWI Historical-only model-fitting contract

**Status:** preregistered design only; do not treat as a completed run or a future SSP model evaluation.

## Immutable input gate

The required input is an artifact from successful v4.5 (not yet validated at time of authoring):
`data/processed/e1_cmip6_v45_awi_historical.npz`, with a matching SHA256 metadata record.

- `tas(t)` predicts `tas(t+1 native day)`, both Kelvin, `[N,1,26,27]`, matched AWI-CM-1-1-MR r1i1p1f1 gn grid.
- Historical **1980–2004** training: 900 pairs, **75 per calendar month**; Historical **2005–2014** validation: 240 pairs, **20 per calendar month**. **No** dates in or after 2015.
- Exactly +24-hour Gregorian-compatible native pairs with ascending unique timestamps, no overlap across train/validation and no extra variables.
- All standardization parameters, monthly P95 gridcell cutoffs, and conditional z reference are computed using training targets **only**.
- Fitting or checkpoint selection must never inspect SSP245/SSP585 temperatures.

## Training stages — gate before expensive runs

1. Use saved v4.5 SHA256 and run an offline 1-seed 2-epoch smoke test for **both FNO and U-Net** with standard MSE; verify finite loss and saved checkpoint reload, and compare with validation Persistence.
2. Only after smoke succeeds, lock **seeds 11/22/33, 100 epochs**, model widths FNO=8/modes=4/depth=2, U-Net width=8, Adam 1e-3, batch size 8, fixed input standardization, and the 3 predeclared training hot multipliers 1×, 2×, 4× (loss tail-weight parameters 0/1/3).
3. Select per-arm checkpoint using **unweighted Historical 2005–2014 validation MSE only**, preserve all arms and all seeds, not just the best. Report validation RMSE, MAE, bias, area-weighted RMSE, locally-defined hot Recall/Precision and forecast hot event fraction, plus parameter counts. Document compute runtime.
4. **Freeze/serialize all checkpoints** and training-only normalization and monthly P95 before any SSP dataset is read. Each model checkpoint must be immutable when future SSP predictions are generated.
5. Later test locked checkpoints on SSP245 and SSP585 *without* any future-target fitting; distinguish 2030/2050/2080 sample-day pilot metrics from full period results. Report weather-year and climate-model ensemble uncertainty, and clarify AWI climate simulation is not an observed future truth dataset.

## Methodological caveats

The 900/240 Historical pairs span 25/10 distinct years but are only **3/2 days per month per year**. A per-cell monthly empirical P95 has 75 training targets (about four tail exceedances/month in expectation). Therefore tail-weighting can be prototyped, but claim neither robust heatwave episode reproduction nor fully representative distribution tails until longer contiguous daily coverage and independent source-model replication. The same-month P95 is fitted across all 25 training years, so long-term changes within the Historical training period are deliberately part of a pooled reference; sensitivity analyses for nonstationarity are still necessary.

These native daily-mean CMIP6 models are **not** directly comparable in absolute RMSE to the earlier ERA5 instantaneous +24h experiment as if they shared one prediction task.
