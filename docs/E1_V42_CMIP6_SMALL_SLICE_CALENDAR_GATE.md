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
