# E1 v5.0 — independent BCC-CSM2-MR native-region data gate

## Why this replication matters

The verified [AWI v4.8 Historical-trained, frozen SSP evaluation](E1_V48_VERIFIED_FROZEN_SSP245_SSP585_RESULTS.md) is restricted to one climate-model realization and two prescribed sample dates/month in 2030, 2050 and 2080. It establishes a useful *within-model* projected-domain forecast benchmark, not a robust climate-model ensemble result. An independent GCM is needed before discussing cross-model stability.

**BCC-CSM2-MR r1i1p1f1 gn** was found in the same verified Pangeo CMIP6 metadata catalog, and its real native historical, SSP245 and SSP585 daily `tas` Zarr stores were confirmed accessible in [v4.9](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45865322183). Exact first/last CF dates: Historical 1850–2014; both SSPs 2015–2100, with native Kelvin units. **No BCC model fit or forecast score yet.**

## v5.0 bounded source-target audit

[One-time real BCC native regional pilot workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868440215). It reads source/target pairs at 2010 Jan 15 and Jul 15 for Historical, and 2030 Jan 15 and 2050 Jul 15 for each SSP. The region is 25–50°N and 235–260°E (125–100°W).

Checks are: strict catalog model/member/grid identity; exact native CF calendar +1 day; Kelvin and finite values in every selected native regional cell; geographic-coordinate grid actual-value checks (not just equal dimensions); no global temperature field reads or model training. Number of native grid cells is **not assumed equal to AWI's 26×27**. All six source/target regional pairs must pass before training-oriented BCC extraction. The pilot Persistence numbers are only data-pipeline sanity checks.

The frozen source-identity manifest is `experiments/cmip6_v49_bcc_candidate_catalog.json`, with implementation `scripts/audit_cmip6_bcc_regional_pairs.py` and safeguards `tests/test_cmip6_bcc_regional_pairs.py`.

## Conditional v5.1/v5.2 path

1. If v5.0 succeeds, design BCC Historical-only 1980–2004 (train) and 2005–2014 (validation) daily-mean +1-day paired extraction using BCC's **own** grid and calendar, freeze source/target timing and native longitude alignment.
2. Test monthly P95 fit on training **only**, with an adequate number of observations per calendar month; no AWI normalization, P95 or checkpoint re-use. Document any change to sample dates.
3. Train BCC FNO, U-Net and Persistence under matched, predeclared 1×/2×/4× weighting and seeds 11/22/33, with checkpoints chosen only on BCC Historical validation.
4. Freeze all BCC checkpoints and evaluate SSP245/SSP585 2030/2050/2080 predeclared samples once, without future fitting.
5. Compare architecture/weight **effect directions across models**, not merely pooled correlated gridcell score totals, and describe model structural uncertainty and sampled-date selection constraints.

The status of v5.0 is unverified until its real-data CI run and JSON artifact have passed. Avoid presenting a hypothetical BCC skill result as verified.
