# E1 v1.5 — data coverage preflight for the contiguous-year benchmark

The last CI repair has passed: [Unit tests run 45282547649](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45282547649).

The 2008–2012 real-data workflow is **manual-only** and has not completed a contiguous-year benchmark. The 2010/2015/2020 72-pair pilot is still the only completed ERA5 model experiment. Do not confuse successful unit tests with a completed ERA5 benchmark.

## New pre-training gate

`src/climate_shift/coverage_gate.py` is invoked after data export and the SHA-256 / timestamp audit, before any model fitting. It requires:

- The exact declared sequence of five expected years, with no missing actual years.
- All twelve months in every year, and at least 20 valid 24-hour pairs per month.
- No skipped months in the exporter metadata.
- Kelvin `2m_temperature`, versioned `uniform_across_month_valid_pairs` extraction, correct 24-hour forecast horizon and source/target timestamps proven from actual ERA5 coordinates.
- Matching dataset SHA-256, target timestamps, and source metadata.

It writes `outputs/e1_contiguous_dev_coverage.json` on passing and fails closed on missing/invalid data; GitHub Actions stores diagnostics as artifacts.

## How to run

The manual benchmark is at [E1 contiguous-year development benchmark](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/workflows/e1-contiguous-dev.yml). Select **Run workflow** on the updated main branch. No old failed run should be re-run because old jobs retain their former commit.

This is still a restricted engineering benchmark: up to 24 deterministic daily input samples per month, 2008–2010 train, 2011 validation, 2012 test, five epochs, one seed. It is **not** representative of full ERA5 temporal dependence or climate-change generalization. For formal work, increase coverage, document a fixed split, add explicit compute-matched tuning and area-weighted estimates, and use an independent warmer-climate evaluation.

Do not claim results until the real workflow completes and its artifacts have been reviewed.
