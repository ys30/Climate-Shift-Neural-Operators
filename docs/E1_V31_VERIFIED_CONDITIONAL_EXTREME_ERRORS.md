# E1 v3.1 — Verified conditional temperature-error results (real ERA5)

- **Successful run**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45715908140
- **Downloadable output artifact**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45715908140/artifacts/15070670759
- **Source commit**: `e9dbcf46601eb8aae94622d1e86091a4dea80f34`.
- Verified 2008–2010 **training-only** local/monthly mean and SD; 2011 validation checkpoints; 2012–2014 independent held-out years. Three model seeds (11, 22, 33), 100 epochs. WB2 ERA5 2m temperature, +24h target and 17×17 western-US latitude-weighted grid.
- The anomaly bins below use **observed target temperature standardized against training-only monthly gridcell climatology**. They are **conditional evaluations**, not an actionable test-time predictor input. They do not condition training on held-out targets.
- For bins with multiple seeds, each RMSE/Bias is first computed by seed, then averaged, so rounding and nonlinear aggregation prevent simple arithmetic reconstruction of unstratified RMSE.

## Area-weighted RMSE conditioned on observed training-relative anomaly (K)

| Year | Bin | Persistence | FNO | U-Net |
|---|---|---:|---:|---:|
| 2012 | normal −1≤z≤1 | 3.301 | 2.491 | 2.469 |
| 2012 | extreme hot z>2 | **3.128** | 4.292 | 4.263 |
| 2013 | normal −1≤z≤1 | 3.032 | 2.323 | 2.326 |
| 2013 | extreme hot z>2 | **2.272** | 2.636 | 2.873 |
| 2014 | normal −1≤z≤1 | 3.555 | 2.667 | 2.654 |
| 2014 | extreme hot z>2 | **1.419** | 1.765 | 1.817 |

## Conditional signed error (forecast minus observed, K), extreme hot z>2

| Test year | Persistence bias | FNO bias | U-Net bias |
|---|---:|---:|---:|
| 2012 | −1.869 | −2.894 | −2.954 |
| 2013 | −1.295 | −1.793 | −2.106 |
| 2014 | −0.514 | −0.899 | −1.053 |

## Frequency of high-anomaly gridcell observations in each test year

- 2012: 3.319%; 2013: 2.248%; 2014: 8.738% (cos(latitude)-weighted, train-relative z>2).
- The conditional RMSE over the hot bin is higher for **both neural models** than Persistence in each year, despite substantially lower **all-gridcell** RMSE in those years. This is stronger direct *descriptive* support for a mean–upper-tail skill tradeoff than an aggregate hot Recall difference alone.
- The test-target-defined `z>2` anomaly bin and the grid-local train-only monthly `P95` Recall threshold are two separate definitions. Neither is an independently verified multi-day heatwave event.

## Scientific limits and proposed intervention

1. Three training seeds and three test years do not establish population-wide or future-climate effect sizes; seed variability and meteorological year variability must be distinguished.
2. Yearly 2014 extreme-hot RMSE for Persistence is low relative to 2012, so the >+2σ bin is not equal-difficulty across meteorological years.
3. **No claim that FNO is uniquely affected** is warranted: the U-Net baseline has a similar tail deficit. These compact architectures were MSE-trained.
4. Next controlled causal-method experiment: identical fixed 2008–2010 training and 2011 validation; compare ordinary MSE to training-referenced tail-weighted MSE (plus optional residual forecasts). Freeze training-only P95 and climatological moments; evaluate all-gridcell RMSE *and* conditional z>2 RMSE and P95 Recall on 2012–2014. Fix candidate loss weight and select checkpoint using 2011 validation only. Cost and random initialization must be disclosed.
5. Before drawing inferential conclusions, use day/weather-block resampling and additional independent years, not iid gridcell p-values.

The analyzed results were read directly from the 3 seed JSON files archived in the successful GitHub artifact.
