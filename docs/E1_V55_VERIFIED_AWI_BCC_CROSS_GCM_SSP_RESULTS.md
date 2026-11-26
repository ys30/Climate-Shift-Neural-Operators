# E1 v5.5 — VERIFIED two-GCM Historical → SSP neural generalization comparison

## Provenance and independence gates

- Independent AWI frozen Historical-trained future target evaluation: [v4.8 successful Actions](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174) and [original JSON](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174/artifacts/15140346151).
- Independent BCC frozen Historical-trained future target evaluation: [v5.4 successful Actions](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45872811683) and [original JSON](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45872811683/artifacts/15144315722).
- The **[v5.5 cross-GCM contrast workflow succeeded](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45873362850)**, verifying its 96 records in the [frozen contrast artifact](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45873362850/artifacts/15144640391).
- **AWI-CM-1-1-MR** r1i1p1f1 gn, native `proleptic_gregorian`, Western-US **26×27** grid; **BCC-CSM2-MR** r1i1p1f1 gn, native `noleap`, **23×23** grid. Both daily mean Kelvin `tas(t) → tas(t+1 native day)`. Both Historical-only train 1980–2004, Historical validation 2005–2014, source-specific normalization and gridcell-local monthly P95 fitted **train-only**; 100 epochs, seeds 11/22/33, FNO vs U-Net and 1×/2×/4× hot training multipliers.
- Future data are **CMIP6 GCM-simulated SSP245/SSP585 projections**, *not independently observed future weather*. Each SSP comprises exactly **72 sparse source-target pairs** from 2030/2050/2080, days 8 and 20 each month. Every checkpoint was frozen before future scoring.

## Core replicated observation: 1× FNO beats its own Persistence baseline

| Projected SSP | AWI Persistence RMSE K | AWI FNO 1× RMSE K | AWI FNO improvement | BCC Persistence RMSE K | BCC FNO 1× RMSE K | BCC FNO improvement |
|---|---:|---:|---:|---:|---:|---:|
| SSP245 | 2.1711 | 1.9323 | **11.00%** | 2.8041 | 2.5122 | **10.41%** |
| SSP585 | 2.1188 | 1.8836 | **11.10%** | 2.7419 | 2.4300 | **11.38%** |

These are **within-GCM relative skill contrasts**. Do not pool the absolute AWI and BCC errors or their spatial pixels to claim an external population mean. The direction of FNO improvement over Persistence repeats across two independently trained GCMs and both sampled SSPs, but the two source models and three future years per scenario are too few to support ensemble-level significance.

## Tail-weight response: FNO Recall improves at 4× across both source GCMs

| GCM | SSP | FNO 1× Recall | FNO 4× Recall | Difference (percentage points) | FNO RMSE 1× → 4× K |
|---|---|---:|---:|---:|---|
| AWI | SSP245 | 0.6786 | 0.7084 | **+2.98** | 1.9323 → 1.9599 |
| AWI | SSP585 | 0.7633 | 0.7863 | **+2.30** | 1.8836 → 1.9049 |
| BCC | SSP245 | 0.5716 | 0.6410 | **+6.94** | 2.5122 → 2.5247 |
| BCC | SSP585 | 0.7717 | 0.8132 | **+4.15** | 2.4300 → 2.4419 |

**Direction replicated in 4/4 GCM×SSP pairs:** strong 4× tail weight improves FNO Recall and worsens its mean RMSE relative to FNO standard MSE. This is a **descriptive effect-direction pattern**, not a formal inferential claim: neither GCM pixels nor three seeds are independent CMIP6 climate-model replications. The 2× FNO arm has variable tradeoffs, and U-Net response is not universally monotonic; for BCC SSP585, U-Net strong weighting actually **decreases** Recall from about 0.6822 to 0.6590.

## Critical competing baseline: Persistence has *higher hot Recall* in all four pairs

| GCM | SSP | Persistence Recall | FNO 1× Recall | FNO 4× Recall |
|---|---|---:|---:|---:|
| AWI | SSP245 | 0.7286 | 0.6786 | 0.7084 |
| AWI | SSP585 | 0.7962 | 0.7633 | 0.7863 |
| BCC | SSP245 | 0.7278 | 0.5716 | 0.6410 |
| BCC | SSP585 | 0.8603 | 0.7717 | 0.8132 |

These high exceedance rates are relative to a **fixed model-specific Historical-trained monthly local P95**, not quantiles refitted to projected SSP. Under future warming, this is a shifting-class-prevalence forecasting problem. A raw high event fraction, event Recall, precision, and future warming bias are distinct from daily mean temperature RMSE. **No FNO arm beats Persistence on hot Recall in these four tests**, despite FNO outperforming Persistence on mean RMSE. Include this caveat prominently in the abstract and paper discussion rather than claiming a general neural-operator dominance.

## Scenario/year heterogeneity and non-monotonic sampled prevalence

The BCC SSP245 sampled reference hot-event fraction, using BCC Historical trained P95, was **0.1719 (2030), 0.2103 (2050), 0.1620 (2080)**, *not* monotonic. BCC SSP585 was **0.2880 (2030), 0.2377 (2050), 0.5142 (2080)**, also not monotonic. The 2080 SSP585 fraction is high, but a single model member with two dates/month/selected year cannot separate forced warming from natural climate variability or sampling noise.

AWI SSP245 hot fractions were approx 0.1198, 0.1950, 0.3241 and AWI SSP585 0.1489, 0.2054, 0.4918 for those three years (see AWI v4.8 report). It would be inaccurate to claim every GCM or scenario shows a smooth monotonic increase in sampled high-temperature fraction.

## What the results do and do not establish

**Established:**
- Valid source-specific native-calendar training, native 1-day forecast-target pairing, trained checkpoints and frozen Historical-only preprocessing for two GCMs.
- Both GCMs show positive mean RMSE skill of FNO standard MSE against their own Persistence baseline on both predeclared sparse simulated SSP tests.
- Within both FNO models, strong 4× Historical high-temperature loss weighting improves future SSP hot Recall at the expense of mean RMSE.
- Persistence retains higher high-temperature Recall across all four compared SSP×GCM cells; no blanket neural-win claim.

**Not yet established:**
- Spatial or climate-ensemble independent statistical significance; source-model and member count remains only two GCMs, one realization each.
- SSP-wide, continuous daily or heatwave-duration skill, because dates are sparse and fixed.
- Preservation of **warming amplitude** (e.g., correct 2050–2080 minus Historical temperature-shift signal) by each neural predictor; higher hot exceedance Recall alone does not directly prove climate-change-signal fidelity.
- Climate projection truth or transfer to observed climate states.
- Whether performance persists under more GCMs, different members, denser sampling and new untouched future years.

## Next no-leakage scientific work

1. Preserve v4.8 and v5.4 as immutable first-domain future tests; no SSP-based refitting, weighting or model selection. Document that future sample dates were previously used for data QA.
2. Report paired seed-effect directions and variation by **sampled year**, not false gridcell-level p-values; treat GCM as primary source-model replication unit.
3. Design a **new**, explicitly versioned warming-amplitude / bias diagnostic using fully frozen models, physically grounded Kelvin differences and Historical train climatology. Do not label this as out-of-sample hyperparameter selection.
4. Add at least one additional independent GCM/member and/or a denser fully separate future test period before high-confidence paper claims.

**This is an initial two-GCM simulated future-domain replication, not an observation-validated or publication-ready full CMIP6 ensemble benchmark.**


## Post-aggregation disaggregation check: FNO 1× versus Persistence in each sampled future year

The **frozen v5.5 archived 96-row contrast JSON**, not new parameter fitting, shows that standard-MSE FNO has lower RMSE than each source GCM's Persistence benchmark in **all 12** separately scored GCM × SSP × sampled-year cells:

| GCM | SSP | 2030 within-GCM RMSE reduction | 2050 reduction | 2080 reduction |
|---|---|---:|---:|---:|
| AWI | SSP245 | 8.30% | 9.18% | 16.49% |
| AWI | SSP585 | 11.97% | 5.65% | 15.40% |
| BCC | SSP245 | 7.68% | 14.39% | 8.61% |
| BCC | SSP585 | 10.74% | 15.11% | 9.12% |

This strengthens the **descriptive consistency within the predeclared sparse sample** beyond reporting only aggregate SSP scores, but cannot be called 12 independent climate-ensemble replications or a statistical significance result: the years are sampled within two correlated scenario trajectories, each with only 24 fixed source-target dates, and neural random seeds share training data. Year-level improvement percentages differ, supporting explicit reporting of time and GCM heterogeneity rather than a single universal number. No model training or score-based checkpoint selection occurred during this additional aggregation.
