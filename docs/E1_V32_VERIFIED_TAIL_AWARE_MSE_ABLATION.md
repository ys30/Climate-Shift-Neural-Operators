# E1 v3.2 — VERIFIED paired ERA5 Tail-aware Loss intervention

**Success run:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45718431425  
**Full artifact:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45718431425/artifacts/15071726852  
**Run commit:** `280b38108cb4bf0015cc2030149098dc0804cf8d`.  
Both arms, their three independent training seeds, 100-epoch histories, real ERA5 WB2 extraction, strict temporal isolation, test-year metrics, anomaly-bin diagnostics and artifact archiving passed.

**Design**: 2008–2010 train, 2011 validation, 2012–2014 held-out, 24 24h pairs/month, same 17×17 grid, FNO width8/modes4 and U-Net width8. Both train 100 epochs with Adam and use unweighted **2011 validation MSE** to choose best checkpoint. Control = unweighted MSE; treatment = per-gridcell training loss `MSE * (1+3*[train-target temperature > train-only month/gridcell P95])`. Thus training hot observations have multiplier 4, all others multiplier 1. No heldout data enter the P95 reference or checkpoint policy. Results refer to fixed three training seeds 11,22,33, not independent meteorological ensembles.

## Real three-seed pooled model scores

| Model | Arm | cos(lat)-weighted RMSE (K) | Monthly grid-local P95 recall | P95 precision | Mean predicted hot-gridcell fraction | P95 true hot-gridcell fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FNO | MSE | 2.858615 | 0.503495 | 0.630563 | 0.092054 | 0.115272 |
| FNO | Tail-aware weight 3 | 2.875900 | 0.544448 | 0.608853 | 0.103220 | 0.115272 |
| U-Net | MSE | 2.840760 | 0.457515 | 0.663722 | 0.079600 | 0.115272 |
| U-Net | Tail-aware weight 3 | 2.860615 | 0.543282 | 0.614217 | 0.102661 | 0.115272 |

Changes in three-seed mean: FNO **+0.017285 K** overall RMSE, **+4.095 percentage points** local Recall, **−2.171 pp** local Precision; U-Net **+0.019855 K** RMSE, **+8.577 pp** Recall, **−4.951 pp** Precision. Both arms underpredict the hot-gridcell frequency compared to truth, though treatment reduces this negative frequency bias. This is an explicit **Precision–Recall** tradeoff, not a free increase in accuracy.

Seed-paired RMSE changes (treatment minus control, K): FNO `[+0.021013, −0.007360, +0.038202]`, U-Net `[−0.023331, +0.019287, +0.063609]`. Seed-paired Recall changes: FNO `[+0.016878, +0.036567, +0.069414]`, U-Net `[+0.101191, +0.156469, −0.000358]`. In particular, **the U-Net Recall gain is not positive for every seed**.

## Per held-out year: extreme-hot z>+2 conditional weighted RMSE (K)

| Year | FNO MSE | FNO Tail-aware | U-Net MSE | U-Net Tail-aware |
| --- | ---: | ---: | ---: | ---: |
| 2012 | 4.2921 | **4.0826** | 4.2630 | **3.9218** |
| 2013 | 2.6363 | **2.3991** | 2.8729 | **2.4855** |
| 2014 | 1.7646 | **1.6240** | 1.8174 | **1.6753** |

## Per held-out year: extreme-hot z>+2 conditional signed bias (K; prediction − truth)

| Year | FNO MSE | FNO Tail-aware | U-Net MSE | U-Net Tail-aware |
| --- | ---: | ---: | ---: | ---: |
| 2012 | −2.8943 | **−2.6389** | −2.9542 | **−2.5241** |
| 2013 | −1.7933 | **−1.5243** | −2.1062 | **−1.5793** |
| 2014 | −0.8990 | **−0.6776** | −1.0535 | **−0.6569** |

These are model-seed averaged conditional metrics. The hot anomaly bin (observed z>2) is a **different condition** from training P95 and test P95 prediction recall. All hot-bin RMSE and signed bias changes favor treatment in this three-year sample, while pooled overall error increases slightly.

## Interpretation / next causal design

1. This is a **controlled optimization intervention**: the independent change is the training target-referenced high-tail loss weight, not new architecture or new data. The result supports that the **training loss** contributes to the error/recall tradeoff in these compact historical models.
2. Do **not** claim dominance across all metrics: Precision drops by approximately 2.2 pp (FNO), 5.0 pp (U-Net), and aggregate RMSE increases slightly on average.
3. Do not claim climate-change OOD improvement: 2012–14 are historical year holdouts and the entire analysis is on sparsely sampled dates.
4. A modest tail-weight treatment (e.g. weight=1, multiplier=2 rather than 4) and selection by 2011 validation **Pareto criteria fixed before the heldout-year test** is scientifically more appropriate than searching over many weights using 2012–14 results.
5. Independent future-climate CMIP6 historical-to-SSP evaluation and day/weather-block uncertainty remain priorities. No future skill or heatwave-duration results have been measured.
