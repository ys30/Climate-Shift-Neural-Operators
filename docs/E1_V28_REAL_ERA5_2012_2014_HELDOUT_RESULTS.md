# E1 v2.8 — verified multi-year heldout ERA5 2012–2014 benchmark

## Verified run evidence

- GitHub Actions successful run: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45664389336
- Artifact: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45664389336/artifacts/15047007665
- Execution commit: `e5be9ebdf26b2206229911be5a6b3c5ab2f9646b`.
- Train frozen at **2008–2010** (864 examples), validation at **2011** (288), test held out for **2012, 2013 and 2014** (288 each, 864 combined). Source is true WeatherBench 2 ERA5 paired 2-m temperatures over the western U.S., 17×17, 24-hour lead, with timestamp audit and actual grid latitude weighting. 24 samples/month/year: sparse source sampling, **not full continuous historical days**.
- Independent model training seeds 11, 22, 33; 100 epochs; validation-based best checkpoint. FNO width 8, modes 4, depth 2; U-Net width 8.
- All jobs passed: environment, targeted unit tests, data extraction, month coverage, year-isolation audit, all training seeds, per-year output integrity and artifact upload.

## 2012–2014 pooled observed results

| Model | Area-weighted RMSE (K), 3-seed average | Latitude-weighted train-only monthly gridcell P95 Recall |
| --- | ---: | ---: |
| Persistence | 3.513108 | **0.606759** |
| FNO | 2.858615 | 0.503495 |
| U-Net | **2.840760** | 0.457515 |

Relative mean weighted RMSE reduction vs Persistence: **18.63% FNO** and **19.14% U-Net**. Trained neural models have lower local extreme recall than Persistence on the pooled three-year holdout.

## Interpretation boundaries

1. Pooling 2012–2014 supports a broader historical weather-year test than 2012 alone; nevertheless it does not establish **every test year** improves mean skill or loses extreme recall. The existing per-seed JSON artifacts already hold yearly metrics; a later aggregation adds a consolidated year-by-year, across-seed summary.
2. Three initialization seeds on fixed weather samples do not create three independent climate experiments. Do not treat seed SD as a weather-year uncertainty interval.
3. Neither the current sparse monthly sample nor locally fitted thresholds establish multi-day heatwave event identification.
4. This benchmark is still an **in-distribution historical weather forecast across years**, not verified generalization under systematic future climate warming, OOD spatial transfer or CMIP6 SSP future conditions.
5. No conclusion of U-Net architecture superiority follows from a ~0.018 K pooled difference without broader tuning, compute-matched budgets and independent years.

Next numerical gate: complete the [year-by-year aggregate rerun](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45679648002) to report actual 2012/2013/2014 scores separately rather than interpreting pooled results as uniform annual behavior.
