# E1 v1.8 — observed three-seed ERA5 replication (unweighted)

**Run**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45295324701  
**Artifact**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45295324701/artifacts/14875892686  
**Commit tested**: `583fbc9556e628390d060a71d13328fd398e0bed` (v1.6, **before** observed-latitude weighting was implemented).  
**Run status**: SUCCESS — unit tests, actual WeatherBench 2 ERA5 data extraction, strict monthly coverage and +24h-origin/time partition audits, FNO/U-Net/persistence comparison, artifact upload.

## Scope

Exactly 1,440 sampled 17×17 temperature (K) source–target pairs for 2008–2012; 24 equally spaced valid pairs/month. Train=2008–2010 (864), validation=2011 (288), test=2012 (288). All three models evaluated the same 2012 test split; neural nets trained for 5 epochs each, seed 11/22/33. Reported results are **unweighted grid-cell** metrics.

ERA5 dataset SHA-256: `f781bc873630e9094a8d4704f968e565c01b44716e5b934a829d39f102c8913b`, the same file content as the prior one-seed run; separate ERA5 reconstructions can differ at the byte level even if arrays coincide, so check each artifact.

## Actual results

| Model | RMSE mean ± between-seed SD (K) | MAE mean ± between-seed SD (K) | RMSE per seed (11,22,33) |
| --- | ---: | ---: | --- |
| Persistence | 3.536877 ± 0 | 2.389464 ± 0 | 3.536877, 3.536877, 3.536877 |
| FNO | 3.244366 ± 0.021658 | 2.356423 ± 0.012452 | 3.223109, 3.243588, 3.266403 |
| U-Net | 3.189295 ± 0.020785 | 2.334553 ± 0.017940 | 3.187384, 3.169532, 3.210969 |

| Model | Hot-gridcell recall by seed (11,22,33) | Mean |
| --- | --- | ---: |
| Persistence | 0.732859, 0.732859, 0.732859 | 0.732859 |
| FNO | 0.513311, 0.563640, 0.558169 | 0.545040 |
| U-Net | 0.532093, 0.514406, 0.458607 | 0.501702 |

The threshold is the training-data **global gridded P95**, about 305.7829 K. These are **hot grid-cell exceedances**, not contiguous multi-day heatwave events.

## Interpretation

- U-Net beats FNO on test RMSE for each of the three observed seed pairs, contrary to the earlier single-seed ranking. This is **not proof of superior U-Net architecture** because only three seeds, a small dataset, short training and unequal model parameter/compute budgets were used.
- Both trainable models have lower global RMSE than persistence but consistently lower high-temperature *recall* at the fixed global training-P95 threshold. This creates a concrete future-study hypothesis: improved mean predictive skill can coexist with an extreme-fidelity penalty.
- Across-seed SD quantifies initialization/training sensitivity, **not sampling uncertainty across independent weather events or years**. Do not use it as a confidence interval for climate risk or warming response.
- The v1.7 code adding cos(latitude) area weighting was pushed **after** the v1.6 workflow used here. Consequently all figures in this document are **unweighted**. A fresh run on v1.7+ code is required for area-weighted metrics.

## Next evidence requirements

1. Re-run the manual three-seed workflow on updated main to obtain truly **area-weighted** overall and hot-tail scores; preserve both unweighted and weighted columns.
2. Record *paired differences* from each seed versus persistence and FNO versus U-Net rather than treating 3 seeds as independent weather replicates.
3. Build seasonal/local climatology hot thresholds on training-only data and assess blocked time uncertainty across independent test years.
4. Run a separate CMIP6 climate-shift experiment. No warmer-climate generalization claim can be made from this 2012 historical test.
