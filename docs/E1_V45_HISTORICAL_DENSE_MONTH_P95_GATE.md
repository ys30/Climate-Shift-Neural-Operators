# E1 v4.4 VERIFIED; v4.5 25-year AWI Historical fit-data gate

## 1. v4.4 completed evidence

- **Success:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45854938557
- Derived JSON artifact: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45854938557/artifacts/15135517585
- The fixed cohort of **312 real AWI-CM-1-1-MR r1i1p1f1 gn Western-US native daily tas +1-day** pair reads fully passed. The source and target grids were 26×27 and had finite, physically plausible Kelvin values at every sampled gridcell. All cases used exact calendar-native +1 day.
- Historical: **168 pairs** (96 training-prototype, 72 validation-prototype), equal 14/month across all historical folds. SSP245: **72 pairs**, equal 6/month; SSP585: **72 pairs**, equal 6/month.
- Descriptive pooled Persistence cosine-latitude RMSE on these selected sampled pairs: Historical **2.342443 K**; SSP245 **2.171081 K**; SSP585 **2.118762 K**. These are **not** trained neural forecast scores or representative full SSP-year skill statistics.
- No scenario test fields enter the training/validation group. No climate-change attribution or future skill is established.

## 2. Why the preceding 4-year training prototype is not enough

The 1980/1990/2000/2004 training prototype read two dates/month/year = **8 training targets per calendar month**. Our existing exploratory training-local month/gridcell P95 threshold requires **at least 20 training targets per calendar month**. Using the old threshold function on only 8/month is invalid under our predeclared guard. Four selected training years would also underrepresent long-term weather variability.

## 3. v4.5 upgraded Historical-only fit-data protocol (launched, not verified until finished)

**Current one-time workflow:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857281051

Historical experiment ONLY for model-member `AWI-CM-1-1-MR/r1i1p1f1/gn`, variable daily `tas` in K, 26×27 native Western-US regional grid (25–50°N, 235–260°E); every pair is `tas(t) → tas(t+1 native day)`. For this one model the verified native calendar is proleptic Gregorian, permitting ISO `datetime64` timestamps; **do not reuse** this exporter unmodified on 360_day models.

| Partition | Historical years | Fixed per-month source days | Count | Target counts/month |
|---|---|---|---:|---:|
| Training only | **1980–2004** inclusive (25 distinct years) | **6, 16, 26** | **900** | **75** |
| Validation only | **2005–2014** inclusive (10 distinct years) | **8, 20** | **240** | **20** |
| Total | | | **1,140** | |

No SSP245/SSP585 data will be opened or added to the v4.5 Historical NPZ. This cleanly prevents inadvertent future SSP normalization, tail threshold fitting and checkpoint selection. The original v4.4 sparse future scenario audit stays independently archived; it is not silently added to model fitting.

### End-to-end strict gates

1. Require exact source/catalog model-member-grid and `historical/day/tas`; fail if SSP source path appears.
2. Require immutable 25-year/10-year split and date schedule, 1,140 distinct forecast origin/target pairs, exact native +1 day lead and strictly separated source/target bounds.
3. Verify finite regional Kelvin values, shape 26×27, grid latitudes and source/target time alignment for **every** selected pair.
4. Save reproducible compressed regional NPZ with `X, y, years, source_times, target_times, latitude`, plus JSON with SHA256 input hash and schedule hash.
5. After saving, run the repository's strict `load_npz_pairs` input validator and refit monthly grid-local P95 on **900 training targets only**, verify threshold shape 12×26×27 and **75 samples per calendar month**.
6. Report descriptive Historical validation-only Persistence area-weighted RMSE as a sanity check; **no** trained model skill yet.
7. **Do not call the existing train/val/test `compare_e1.py` as-is:** it requires a third heldout group in a single NPZ and would conflate Historical with future SSP if fed a blended dataset. Separate explicit Historical train/validation fit and a read-only SSP evaluation harness will be needed.
8. Archive the derived Historical regional NPZ plus source/provenance JSON for subsequent model fitting; the workflow is bounded at 70 minutes and should not be automatically retriggered by unrelated pushes.

### Interpretation caveats

- **Three fixed sampled days per month per training year are not continuous daily trajectories.** These can support small controlled 1-day regression experiments but not heatwave episode length, persistence, or continuous day-level extremes.
- 75 targets per month exceed the technical minimum of 20, but the monthly P95 still has limited tail effective sample size; report this honestly. Grid cells/adjacent dates are not independent climate realization replicates.
- Climate-scenario projection skill and warming-signal retention are *not yet measured*; do not infer climate-change resilience from Historical-only validation.
- The daily CMIP6 mean-temperature target cannot be directly compared as an identical task to the existing ERA5 instantaneous +24h prediction.


## VERIFIED v4.5 completion and immutable source contract

**End-to-end success:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857281051  
**Data artifact:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857281051/artifacts/15138645070

The workflow successfully extracted 1,140 real Historical daily `tas(t)→tas(t+1 day)` regional pairs, verified source/target lead and train–validation time isolation, and passed the NPZ strict input round-trip and train-only monthly local P95 fit. Actual train=900 (25 years; 75 per month), Historical val=240 (10 years; 20 per month), shape 26×27, all finite Kelvin. The Historical validation-only Persistence weighted RMSE was **2.3297778468 K**.

**Frozen NPZ SHA256:** `ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914`. This exact hash is mandatory for the next training phase and prevents accidental use of SSP data or a changed test cohort.

The separately committed `cftime` dependency corrected prior default CI failures; [default Unit tests](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857475991) and [synthetic model smoke](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857476042) both completed successfully.

## v4.6 one-shot Historical-only FNO/U-Net smoke launch (pending outcome)

**Run:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45859545722. It downloads that verified v4.5 artifact rather than re-fetching CMIP6 data, rejects a checksum mismatch, fits the two fixed model architectures on Historical training with standard MSE for **two epochs, seed 11**, and saves check-point state dictionaries with frozen normalization and training-only monthly P95 thresholds. Checkpoint choice uses Historical 2005–2014 validation only. No SSP store is opened, and no future-climate forecast or trained SSP metric is claimed unless a future independent evaluation is actually executed. Scientific protocol: [v4.6 blinded fitting](E1_V46_HISTORICAL_FIT_AND_SSP_BLIND_TEST_PROTOCOL.md).
