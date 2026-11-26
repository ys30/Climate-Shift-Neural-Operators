# E1 v4.4 — AWI-CMIP6 sparse multiyear cohort: frozen scope and live audit

## Objective and evidence boundary

The v4.3 [real 26×27 regional +1-native-day temperature pairing workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764223116) succeeded on six individual dates. This v4.4 audit expands that **data pipeline integrity check** to 312 predeclared pairs spanning multiple Historical and projected years. This is not model training, generalization evidence or continuous-day coverage.

**Live one-time real data workflow**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45854607704

## Preregistered schedule

| Source experiment | Cohort | Actual sampled years | Exact days in each month | Expected pairs |
| --- | --- | --- | --- | ---: |
| Historical | Training prototype | 1980, 1990, 2000, 2004 | 8, 20 | 96 |
| Historical | Validation prototype | 2005, 2010, 2014 | 8, 20 | 72 |
| SSP245 | Future-domain audit-only | 2030, 2050, 2080 | 8, 20 | 72 |
| SSP585 | Future-domain audit-only | 2030, 2050, 2080 | 8, 20 | 72 |
| **Total** | | | | **312** |

AWI-CM-1-1-MR ensemble `r1i1p1f1`, native `gn` grid, daily mean `tas` Kelvin, exact `tas(t)→tas(t+1 native calendar day)`. Western-US latitude 25–50°N and longitude 125–100°W (235–260°E); native 26×27 grid. No remapping to ERA5's 17×17 grid.

**Important:** There are only two sample days per month *in each selected year*. These are NOT 25 continuous training years, and the prototype Historical training has **8 observations per calendar month (96 total)**, insufficient for the previous `fit_local_monthly_p95` function requiring **at least 20 samples per month**. **Do not directly reuse the old tail-aware training code on v4.4 sample.** The eventual full benchmark needs adequate training-month coverage and strict source/target temporal isolation, plus native-calendar handling for non-Gregorian CMIP6 models.

## Automatic data checks

- Catalog source identity and exact ensemble/member/grid are frozen and the real Zarr store names are recorded.
- Predeclared schedule is evaluated once: historical train samples precede every historical validation sample; SSP target dates are never used for training, checkpoint selection or threshold fit.
- Every sampled source day has an exact **+1 native day** target and remains within its source experiment and year; no time-boundary contamination.
- Each 26×27 source and target regional Kelvin grid must be finite and physically plausible, and shape/calendar must match across source datasets.
- Month coverage, fold counts, raw-data archiving restriction and deterministic schedule SHA256 are reported.
- Only descriptive single-cohort Persistence weighted RMSE is calculated as a **pipeline sanity check**, not future neural model skill.
- The workflow produces a small derived JSON record. It deliberately does **not** archive full CMIP6 temperature arrays, train FNO/U-Net weights or calculate trained-model SSP scores.

## Go / no-go for next v4.5

Proceed only if **all 312** exact source/target pairs are validated and the workflow succeeds. To train models properly, expand Historical training to **≥20 per training calendar month** (and validate samples), preserve native calendar identity, record fit-only normalization/P95, and separate ScenarioMIP testing without any selection. Do not claim validation results on incomplete sparse data.

See [previous native target protocol](E1_V43_AWI_CMIP6_DAILY_PAIR_REGIONAL_PROTOCOL.md). Historical- and SSP-year forecasts are projections of a climate-model simulation, **not independently observed future climate truth**.
