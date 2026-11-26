# E1 v4.2 — Real CMIP6 tiny-slice and calendar verification gate

## Prior proven data access

The [v4.1 direct public Zarr probe passed](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759128555) for **six real daily tas Zarr stores**, forming two identical source/member/grid trios:

- ACCESS-CM2 / r1i1p1f1 / gn: historical, ssp245, ssp585
- AWI-CM-1-1-MR / r1i1p1f1 / gn: historical, ssp245, ssp585

Every store passed anonymous open, native Kelvin units, time CF metadata presence, latitude/longitude coordinate schemas and matched spatial dimensions across scenarios. See [v4.1 provenance](E1_V41_VERIFIED_CATALOG_AND_ZARR_ACCESS_GATE.md) and [full GitHub artifact](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759128555/artifacts/15089100952).

## The new v4.2 gate

The [corrected tiny-real-data workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759757258) is launched; numerical and calendar findings are **pending successful final log and artifact inspection**.

The probe `scripts/audit_cmip6_small_slices.py` uses the same catalog-grounded six stores to:

- Open actual Zarr arrays anonymously, verify variable and Kelvin metadata (already passed in v4.1).
- Decode first and last native CF dates via **cftime**, retaining model calendars such as `360_day` rather than implicitly forcing Gregorian calendar.
- Check expected Historical ending around 2014 and SSP beginning about 2015, while recording exact dates rather than guessing them.
- Sample at most 3 selected time indices × 2×2 grid cells (= 12 temperature values) **per store**, and record bounded finite/physical Kelvin summaries. Read only four lat/lon edge scalar values per store.
- Check all three scenarios have compatible calendar and spatial grid shape, in addition to matched catalog source/member/grid identifiers.
- Archive the exact pass/fail outcome for *both* candidate model trios; no fabricated success on inaccessible samples.

A companion test module `tests/test_cmip6_small_slices.py` asserts missing future datasets and bad dates are rejected. The first trigger was issued before a standalone/module import compatibility fix; only the **corrected v2 run** should be interpreted as the current audit.

### Still unverified

The small sample cannot establish complete spatial/temporal data coverage, missingness, regional bounds, grid conventions, faithful target pairing or future neural model skill. No CMIP6 trained forecast has yet been produced, and daily `tas` is not a direct replacement for instantaneous ERA5 +24h snapshots.

**Next honest gate:** once real small samples and dates pass, design regional subset extraction with calendar-native day indexing, coverage/missing checks, a historical train/validation split, fixed daily +1 day forecasting target, plus clean baseline and same loss candidates. Only then train FNO/U-Net separately on native CMIP6. Keep SSP targets out of model fitting and checkpoint selection.


## v4.2 actual read results and time-coverage correction

The initial [real-data small-slice job](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759757258) **failed in its final overly restrictive year assertion** after successfully opening and sampling all six stores; this is NOT a climate-model training failure. Its [archived JSON evidence](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759757258/artifacts/15089577174) demonstrates:

| Model | Experiment | Verified first day | Verified last day | Native calendar |
| --- | --- | --- | --- | --- |
| ACCESS-CM2 | historical | 1850-01-01 | 2014-12-31 | proleptic_gregorian |
| ACCESS-CM2 | ssp245 | 2015-01-01 | 2100-12-31 | proleptic_gregorian |
| ACCESS-CM2 | ssp585 | **2201-01-01** | **2300-12-31** | proleptic_gregorian |
| AWI-CM-1-1-MR | historical | 1850-01-01 | 2014-12-31 | proleptic_gregorian |
| AWI-CM-1-1-MR | ssp245 | 2015-01-01 | 2100-12-31 | proleptic_gregorian |
| AWI-CM-1-1-MR | ssp585 | 2015-01-01 | 2100-12-31 | proleptic_gregorian |

Dates are native-calendar time-coordinate values decoded by cftime. Every experiment yielded finite Kelvin temperature readings in a bounded 3-time × 2×2 sample (maximum 12 temperature values per store). **The ACCESS-CM2 SSP585 candidate is an extended-period source, not a valid common 2015–2100 SSP585 source**. Its older catalog-matched version was not checked for an alternative 2015–2100 period; do not infer none exists.

### Correction to eligibility policy

The original small-slice code checked only SSP start >=2014 and thus incorrectly counted ACCESS-CM2 as compatible. We corrected that scientific logic: historical must end in 2014, and **each** SSP must begin in 2015 and extend through at least 2100 with matching native calendar, Kelvin units and grid shape. A dedicated regression test makes `ssp585 2201–2300` ineligible. The original workflow's year-<=2200 assertion was also corrected to **record** extensions through 2300 without treating them as eligible for the chosen experiment.

The [corrected one-time audit run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45760624361) must pass before claiming the stricter gate is operational. If it passes, AWI-CM-1-1-MR is the sole **currently verified candidate** for the common historical/SSP 2015–2100 experiment; ACCESS-CM2 can be reconsidered only after sourcing and auditing a different 2015–2100 SSP585 store.

This is a data eligibility result only, **not a future SSP forecast result**.
