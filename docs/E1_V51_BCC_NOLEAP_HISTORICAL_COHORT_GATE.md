# E1 v5.0 verified; v5.1 BCC Historical-only noleap cohort preflight

## Real source gate: v5.0 SUCCESS

[GitHub Actions BCC true daily regional Kelvin paired-data gate](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868440215) succeeded in all unit, source and postprocessing checks. [Original derived JSON evidence](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868440215/artifacts/15141801157).

- BCC-CSM2-MR `r1i1p1f1 gn`, 25–50°N and 235–260°E Western-US subset, native **23×23 grid (529 cells)** — **not** the AWI 26×27 grid.
- Native calendar is **`noleap` / 365_day**, unlike AWI's proleptic Gregorian; BCC Gregorian datetime64 interpolation would therefore be wrong around leap years.
- Historical real paired dates: 2010-01-15→16 and 2010-07-15→16.
- SSP245 and SSP585 each passed real paired dates: 2030-01-15→16 and 2050-07-15→16. All are one true native CF day apart.
- All six regional temperature pairs were finite and physically plausible Kelvin; the *actual spatial coordinate SHA values* match between BCC Historical and both SSPs, not just their regional dimensions.
- The derived single-pair Persistence scores are **QA only**, not BCC forecast generalization evidence.

## v5.1 source-specific training-data exporter: initiated

[One-shot Historical-only BCC native noleap dataset audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868844059).

Exact preregistered split (same sampled calendar months and year partitions as AWI):

| Source | Years | Sample origin days/month | Number of pairs | Train-only P95 sample count/month |
|---|---|---|---:|---:|
| BCC Historical training | 1980–2004 inclusive | 6,16,26 | 900 | 75 |
| BCC Historical validation | 2005–2014 inclusive | 8,20 | 240 | — |

The extractor verifies all 1,140 original **native noleap** source→target day pairs, finite Kelvin at every 23×23 gridcell and strict Historical train/validation time isolation. The proposed artifact stores `source_cf_day_value`, `target_cf_day_value`, native calendar date strings, `months`, `train_mask`, 23×23 `X/y`, BCC latitude/longitude and **training-only BCC gridcell-month P95 thresholds**; it does not pretend that `noleap` time is Gregorian `datetime64`. Raw SSP data, AWI normalization, AWI monthly P95 and AWI checkpoints do **not** enter this stage.

Only if the workflow and its archived SHA256 pass can subsequent BCC-only FNO/U-Net training start. Reusing the existing `climate_shift.data.load_npz_pairs` (which assumes datetime64 for paired timestamps) on a native `noleap` file is invalid; a native-calendar-aware training/metrics adapter is required. As with AWI, the resulting cohort is sparse 3 days/month/year, not a continuous 25-year daily record.

## Next scientific goals after the BCC gate

- Independent BCC Historical training (architecture family, epochs, seeds and 1×/2×/4× weighting matched to AWI), models/checkpoints trained and normalized **on BCC only**.
- SSP245/SSP585 **future-domain** scoring with frozen BCC checkpoints, once a native-calendar-aware evaluation harness is validated.
- Report **between-GCM** differences and uncertainty, alongside intra-model seed effects; two climate-model members are still not a comprehensive ensemble.
- The verified AWI future-score results remain in [E1 v4.8](E1_V48_VERIFIED_FROZEN_SSP245_SSP585_RESULTS.md), without SSP-based tuning.
