# E1 v1.6 — first successful contiguous-year REAL ERA5 benchmark

**Evidence source:** GitHub Actions [run 45292352961](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45292352961) (2026-11-23 UTC), commit `768627b63711180e86bdfa1f2ecd114bf653bfba`. Workflow outcome: **success**. All unit tests, real-data download, provenance audit, monthly coverage gate, cross-fold timestamp audit, model evaluation and artifact upload completed successfully.

**Archived diagnostics:** [Actions artifact 14874662444](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45292352961/artifacts/14874662444) (`e1-contiguous-dev-diagnostics`).

## Data and exact split

- Source: public WeatherBench 2 ERA5, 2m temperature (K), western-US subset 25–50°N, 125–100°W, 17 × 17 grid, +24-hour point predictions.
- Sampled 2008–2012, **24 equally spaced valid 24h pairs each month**, **288/year**, total **1,440**. Not continuous calendar-day coverage, despite consecutive calendar years.
- **Train 2008–2010:** 864 pairs. **Validation 2011:** 288 pairs. **Test 2012:** 288 pairs.
- Exact +24-hour pair origins independently retained. All 60 year-months present, no skipped months. Strict timestamp isolation passed for train/validation/test; 24h train-to-validation and 24h validation-to-test adjacent-fold gaps.
- Input NPZ SHA-256: `f781bc873630e9094a8d4704f968e565c01b44716e5b934a829d39f102c8913b`.
- Training: epochs=5, batch_size=8, learning rate=0.001, seed=2026, width=8, FNO modes=4; each model selected checkpoint by 2011 validation MSE; training-target-only mean/std.

## Test-set metrics (K)

| Model | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: |
| Persistence | 2.389464 | 3.536877 | +0.084390 |
| U-Net | 2.423087 | 3.264087 | +0.073026 |
| FNO | 2.355280 | 3.234595 | −0.125622 |

FNO's observed RMSE improvement over persistence is **8.55%**, U-Net's **7.71%**, on this specific development test. These are *single-run point estimates*, not statistically reliable model rankings, and differences versus the earlier sparse 72-pair pilot confound training time, data amount, temporal split, and sampling methodology.

## Hot-tail gridcell diagnostic (train-period global P95 = 305.782906 K)

2012 observed exceedance fraction: **6.5888%** across sampled test gridcells. This is a single global scalar, not local-seasonal climatology or a multi-day heatwave event.

| Model | Predicted hot fraction | Hot recall | Hot precision | Conditional hot bias (K) |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 6.5948% | 0.73286 | 0.73219 | −0.83604 |
| FNO | 4.7866% | 0.57312 | 0.78891 | −1.99601 |
| U-Net | 4.1619% | 0.50237 | 0.79532 | −2.56202 |

**Key tension:** both learned models improved all-grid RMSE relative to persistence but **under-predicted the prevalence of hot gridcells**. FNO has higher precision but lower recall than persistence on this test. This result supports studying the *forecast-skill versus extreme-fidelity tradeoff*, but is **not** evidence about climate-change generalization.

## Unresolved requirements

1. At least three independently seeded runs; fixed, identical split and data checksum; consider more than five epochs and fair compute-budget tuning.
2. Area-weighted evaluation with actual latitude coordinates (current metric is unweighted).
3. Continuous 6-hourly/daily time series for heatwave event duration; no duration can be inferred from 24 samples/month.
4. Multiple held-out years / regional OOD designs with blocked uncertainty estimates.
5. A CMIP6 warming-distribution experiment separate from ERA5 calendar-year holdout; no warming-causality conclusion from this test.
6. Check dataset-specific license terms before any ERA5 data redistribution.

**Research status:** a successful **real-data engineering and historical holdout pilot**, not a manuscript-grade claim that neural operators generalize to a warmer climate.
