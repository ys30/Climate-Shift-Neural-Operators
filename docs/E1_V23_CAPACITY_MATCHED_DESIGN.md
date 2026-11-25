# E1 v2.3 — parameter-near capacity experiment design (no training yet)

The active E1 benchmark uses single-variable ERA5 2m-temperature at 17×17 spatial resolution. Its short 5-epoch 2008–2012 benchmark is **not sufficient to compare final FNO versus U-Net capacity**. The [50-epoch manual-only convergence run](../.github/workflows/e1-convergence-dev.yml) is the immediately preceding scientific checkpoint; do not launch a capacity sweep before checking that run's learning curves.

The [capacity planner](../scripts/plan_e1_capacity.py) selects the nearest U-Net width to each proposed FNO configuration **by real scalar degrees of freedom**. Its numbers are analytically verifiable for the existing code: FNO depth 2 with two complex spectral weight banks per layer has `8*w²*m² + 3*w² + 8*w + 1` real degrees of freedom; U-Net parameter counts are computed from the actual `torch.nn.Module` classes, not inferred from depth or FLOPs.

| Proposed FNO width/modes | FNO real scalar DOF | Nearest candidate U-Net width | U-Net real scalar DOF | DOF gap (U-Net minus FNO) |
| --- | ---: | ---: | ---: | ---: |
| 8 / 4 | 8,449 | 9 | 8,488 | +0.46% |
| 12 / 6 | 42,001 | 20 | 40,641 | -3.24% |
| 16 / 8 | 131,969 | 36 | 130,177 | -1.36% |

These are **planned model configurations**, not observed training scores. Counts assume the current code, FNO depth=2 and exactly 1 input/output channel. The currently completed historical experiment used **U-Net width 8**, not the proposed width-9 small *capacity-matched* comparison.

## Planned capacity test procedure

1. Fix identical 2008–2010 train, 2011 validation, 2012 historical holdout, with 2008–2012 ERA5 SHA-256 provenance and verified +24h pairs. Do not select candidate widths based on held-out test scores.
2. Run the manual 50-epoch convergence experiment first, inspect train/validation trajectories for undertraining and overfitting. Select a fixed epoch budget and checkpoint policy from validation only.
3. Run the predeclared Small, Medium, Large FNO/U-Net pairs. Use multiple independent seeds for each, identical data samples, validation-based checkpoint selection and fixed convergence rule. Report compute time and preferably measured device FLOPs or wall-clock throughput; equal parameter DOF does not equal matched compute.
4. Compare **area-weighted RMSE/MAE** and both global train-P95 and month-local train-P95 gridcell recall. Avoid inferring heatwave events from sparse daily samples.
5. Only after model controls, test independent held-out historical years and independent climate-shift scenarios, including CMIP6 controlled warm-season anomalies.

No new real-data training has been executed by this document or planner. The latest observed weighted three-seed results remain in [v2.1](E1_V21_VERIFIED_WEIGHTED_SEASONAL_ERA5_RESULTS.md).
