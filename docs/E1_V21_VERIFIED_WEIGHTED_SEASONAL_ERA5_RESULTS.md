# E1 v2.1 — verified real ERA5 area-weighted, seasonal-local, 3-seed benchmark

## Primary evidence (VERIFIED)

- **GitHub Actions run:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45478166832
- **Run result:** SUCCESS, manual `E1 ERA5 three-seed replication`.
- **Commit used:** `c872bae1edc6eeb8dd308f26967a4f7ae791ff9c`.
- **Artifact:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45478166832/artifacts/14961610828, includes metadata, coverage audit, per-seed metrics and summary.
- **Exact NPZ SHA-256:** `21ba61d319f0732508861418db8c676e3da73b38c59f45749d897fc424a74554`.
- Job steps all passed: test suite, ERA5 extraction, 60-month coverage, independently verified +24h timestamp pairs, three-model training, strict presence checks for area-weighted and month-local metrics, artifact upload.

## Data and design

- Public WB2 ERA5 2-m temperature (Kelvin), 25–50 N and 125–100 W, 17×17 grid, 24-hour prediction horizon.
- **1,440** deterministically selected valid pairs, 24 per month across 2008–2012.
- Train 2008–2010 (864), validation 2011 (288), test 2012 (288); strict chronological separation passed.
- Independent training seeds **11, 22, 33**, 5 epochs, restricted development FNO and U-Net. Persistence is deterministic on fixed input.
- Latitude weighting is **cos(latitude)** on actually recorded source grid rows, not exact cell areas; separate equally weighted gridcell metrics retained.
- Seasonal-local P95: fitted separately for 12 months and each grid cell on training **targets only**; 72 training samples per month per grid cell.

## Actual test results (mean ± between-seed SD)

| Model | Area-weighted RMSE (K) | Area-weighted MAE (K) | Unweighted RMSE (K) |
| --- | ---: | ---: | ---: |
| Persistence | 3.461329 ± 0 | 2.318198 ± 0 | 3.536877 ± 0 |
| FNO | 3.175610 ± 0.022491 | 2.295386 ± 0.013214 | 3.244366 ± 0.021658 |
| U-Net | **3.126368 ± 0.020079** | **2.277568 ± 0.019689** | **3.189295 ± 0.020785** |

Area-weighted RMSE differences (new model minus comparator, lower is better):
- FNO minus Persistence: **−0.285719 K** (about 8.25% lower than Persistence).
- U-Net minus Persistence: **−0.334961 K** (about 9.68% lower).
- FNO minus U-Net: **+0.049242 K** (U-Net is lower in all three training seeds; seed differences +0.024641, +0.068799, +0.054287 K).

## Actual hot-gridcell recall

| Model | Global training P95 recall, latitude-weighted | Monthly-gridcell training P95 recall, latitude-weighted |
| --- | ---: | ---: |
| Persistence | **0.738879** | **0.542995** |
| FNO | 0.556106 ± 0.028895 | 0.425952 ± 0.041343 |
| U-Net | 0.515192 ± 0.038484 | 0.411827 ± 0.051172 |

The month-local thresholds cover all 12 calendar months. Each month has 72 training samples per grid point. The actual fitted threshold range over all months and cells was 266.903972–316.776064 K. A fixed **global** P95 and **monthly grid-local** P95 refer to different event populations; differences between those recall values are NOT causal improvement or degradation from area weighting or seasonal normalization.

## Interpretation with strict limitations

1. **Reproducible historical development signal:** all three seeds produced better mean RMSE for trainable models than Persistence; their high-tail recall was lower under both global and month-local P95 definitions. The observed tradeoff is a hypothesis worth examining.
2. **Model ranking depends on design:** U-Net has lower RMSE than FNO in this small experiment, not evidence that U-Net is inherently superior.
3. **Not independent meteorological replicates:** repeated seeds on the same 2012 holdout do not provide uncertainty over independent weather years or physical extremes. Across-seed SD is not a climatic confidence interval.
4. **Not a full heatwave benchmark:** monthly deterministic sampling is sparse, so consecutive-day heatwave duration or counts cannot be inferred. Three training years are too short for stable climatological P95.
5. **Not evidence of warmer-climate generalization:** ERA5 2008–2012 is a short historical year holdout, not a controlled climate-shift or CMIP6 future experiment.
6. **Not a statistically matched architecture competition:** model sizes, effective compute and hyperparameter tuning were not matched.
7. **Archival distinction:** older real three-seed run 45295324701 used pre-v1.7 code; its results are **unweighted**. Only run 45478166832 has verified area-weighted and seasonal-local outcomes.

## Scientific next gate

Prioritize multi-year blocked holdouts on a longer chronological ERA5 cohort, then controlled temporal-distribution and geography shifts with properly calibrated local extremes. Add residual-prediction or tail-aware learning as an **experimental arm**, not a retroactive alteration to the fixed baseline. Eventually include historical-vs-future CMIP6 controlled climate shifts.
