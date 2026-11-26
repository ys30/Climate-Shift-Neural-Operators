# E1 v3.3 — VERIFIED three-arm tail-aware training sensitivity (historical ERA5)

## Provenance and evaluation scope

- MSE and tail-weight=3 (training hot multiplier 4) controlled paired experiment: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45718431425 (artifact 15071726852)
- Independently run tail-weight=1 (training hot multiplier 2): https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45751577655 (artifact 15086051665); workflow successful
- Matching design: train 2008–2010, validation 2011, test 2012–2014; 24 sampled +24h pairs/month/year; western-US 17×17 real WB2 ERA5 T2m; 100 training epochs, seeds 11, 22, 33. FNO width8 modes4 depth2, U-Net width8; checkpoint selected using **unweighted validation MSE**, same in all arms.
- Model input and local monthly P95 references derived solely from training targets; all arm results computed on the same heldout years. Values below are means over three independent **initialization** seeds, not climate-year ensembles.

## Verified three-arm scores (full 2012–2014 pooled holdout)

| Model | Tail-weight parameter | Hot training multiplier | Weighted RMSE (K) | Local-month P95 recall | Local-month P95 precision | Predicted hot fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FNO | 0 | 1× | **2.858615** | 0.503495 | **0.630563** | 0.092054 |
| FNO | 1 | 2× | 2.872691 | 0.525245 | 0.614977 | 0.098603 |
| FNO | 3 | 4× | 2.875900 | **0.544448** | 0.608853 | 0.103220 |
| U-Net | 0 | 1× | 2.840760 | 0.457515 | **0.663722** | 0.079600 |
| U-Net | 1 | 2× | **2.830819** | 0.476442 | 0.657826 | 0.083684 |
| U-Net | 3 | 4× | 2.860615 | **0.543282** | 0.614217 | 0.102661 |

Truth hot-gridcell fraction is approximately **0.115272** by local monthly train-P95 definition; the forecasted hot fraction rises as weights increase in this sample.

Changes relative to MSE: FNO tail-weight1: weighted RMSE +0.014076 K, Recall +0.021750, Precision −0.015586. U-Net tail-weight1: RMSE **−0.009941 K**, Recall **+0.018926**, Precision −0.005897. No dominance inference: ~0.010 K average RMSE is very small and these three init seeds under the same data do not establish robust generalization.

## Conditional error on observed training-relative extreme-hot anomaly bin z>+2 (K)

| Year | Model | MSE RMSE | Tail-weight 1 RMSE | Tail-weight 3 RMSE | MSE bias | Tail-weight 1 bias | Tail-weight 3 bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2012 | FNO | 4.292 | 4.186 | **4.083** | −2.894 | −2.782 | **−2.639** |
| 2013 | FNO | 2.636 | 2.528 | **2.399** | −1.793 | −1.644 | **−1.524** |
| 2014 | FNO | 1.765 | 1.682 | **1.624** | −0.899 | −0.738 | **−0.678** |
| 2012 | U-Net | 4.263 | 4.044 | **3.922** | −2.954 | −2.711 | **−2.524** |
| 2013 | U-Net | 2.873 | 2.635 | **2.485** | −2.106 | −1.930 | **−1.579** |
| 2014 | U-Net | 1.817 | 1.738 | **1.675** | −1.053 | −0.999 | **−0.657** |

Training weights increase high-target influence and modify the conditional bias; this is not the same as improved **forecast probability calibration**, because these are deterministic regression forecasts and there is no probability-calibrated event head.

## Limitations and next *independent* confirmation

- **Selection bias:** Tail-weight 1 was evaluated after observing the weight-3 heldout results, so 2012–2014 no longer form a completely untouched test set for *choosing between* weights. Do not proclaim any hyperparameter selected on these years prospectively optimal.
- Prepare a final locked protocol with candidate tail weights **0, 1, 3**, identical fixed 2008–2010 train, 2011 validation and no further selection using 2012–2014, then evaluate each on **fresh, previously untouched 2015–2017** WB2 ERA5 heldout years. Report every candidate, not just a winner.
- Training-matched epoch budgets do not imply FLOP/time matched budgets, and the current model is 17×17 one-variable +24h temperature forecasting.
- The >+2σ conditional target bin is defined from observed test targets and is exclusively an evaluation stratification; it must never be used to choose inference-time predictions.
- Historical years are not demonstrated CMIP6 future SSP climate-shift generalization. Day/weather-block uncertainty and independent climate scenarios remain required for climate claims.
