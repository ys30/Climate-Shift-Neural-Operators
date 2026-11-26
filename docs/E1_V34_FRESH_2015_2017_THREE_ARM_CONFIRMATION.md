# E1 v3.4 — Independent 2015–2017 historical confirmation of three tail-loss candidates

## Verified sources and fixed protocol

- **All jobs succeeded**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45754843640
- **Full raw diagnostics artifact (derived JSON, no raw grids)**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45754843640/artifacts/15088091556
- **Run SHA**: `14367ccabc06a2ea88227e340ccd95a6e071bdc9`
- **Input dataset SHA256**: `bd780744a0bb461bf866b0b6aa8072335149fb1815f8991b6dc9856ec959571c`
- Historical WB2 ERA5 2m temperature, western US 17×17, exact +24h input/target; 24 samples per month, 288 each heldout test year. Train 2008–2010 (864); validation 2011 (288); independent heldout 2015/2016/2017 (864). Three initialization seeds **11,22,33**; 100 epochs; fixed model architecture (FNO width8/modes4/depth2, U-Net width8). Loss weights frozen at **0,1,3** (training hot target multiplier **1×,2×,4×**) before the 2015–17 run. Validation checkpoint always chosen by regular unweighted MSE. Training high-tail threshold is fitted to **training-only local monthly gridcell P95**.
- All three arms share the **exact same** underlying input dataset checksum. Test-year results cannot affect training references or checkpoint selection.

## Primary six-arm comparison — three seed averages

| Architecture | Hot training weight | All-gridcell cos(latitude)-weighted RMSE K ↓ | Monthly local-P95 recall ↑ | Precision ↑ | Predicted hot-gridcell fraction |
|---|---:|---:|---:|---:|---:|
| FNO | 1× | **2.795228** | 0.562453 | **0.699256** | 0.141340 |
| FNO | 2× | 2.798838 | 0.598297 | 0.686648 | 0.153262 |
| FNO | 4× | 2.804872 | **0.615854** | 0.682208 | 0.158761 |
| U-Net | 1× | 2.795592 | 0.504122 | **0.741667** | 0.119485 |
| U-Net | 2× | **2.771325** | 0.519567 | 0.737404 | 0.123975 |
| U-Net | 4× | 2.813836 | **0.600419** | 0.695548 | 0.152659 |

## Annual 2015–2017 hot-anomaly RMSE, observed target z>2 (K)

| Year | Architecture | Standard MSE / 1× | Moderate / 2× | Strong / 4× |
|---|---|---:|---:|---:|
| 2015 | FNO | 1.9966 | 1.8879 | **1.8235** |
| 2016 | FNO | 2.2350 | 2.1385 | **2.0790** |
| 2017 | FNO | 2.9944 | 2.8485 | **2.7649** |
| 2015 | U-Net | 2.1172 | 2.0066 | **1.9371** |
| 2016 | U-Net | 2.3857 | 2.2583 | **2.2077** |
| 2017 | U-Net | 3.1030 | 2.8363 | **2.7163** |

## Reproducibility and interpretation

- FNO vs standard MSE: moderate / 2× increases pooled all-gridcell RMSE by **+0.003610 K**, and recall by **+0.035844**; strong / 4× increases RMSE by **+0.009644 K**, and recall by **+0.053401**. Conditional hot RMSE improves for **each** year at both strengths.
- U-Net moderate / 2×: weighted RMSE improves by **−0.024267 K** (~0.87% reduction), local recall increases **+0.015445**, precision decreases **−0.004263**, and hot-anomaly RMSE improves in all three heldout years. Crucially **all three initialization seeds** improve total RMSE (paired changes **−0.025240, −0.007975, −0.039585 K**), but **not all three** improve recall (paired changes **−0.003132, +0.002294, +0.047175**).
- U-Net strong / 4×: recall rises by **+0.096297** but overall RMSE worsens by **+0.018245 K** and precision falls by **−0.046119**. Strong treatment is useful for higher recall but does **not** dominate moderate treatment on the mean-skill/precision axis.
- Original 2012–2014 exploration, separately documented in [v3.3](E1_V33_VERIFIED_THREE_WEIGHT_PARETO_RESULTS.md), showed a similar small moderate U-Net mean-skill and local-recall improvement, but those years were involved in choosing candidate weights. 2015–2017 is the new confirmation set for the **frozen full candidate slate**; no post hoc hyperparameter tuning from it is justified.
- Three seeds and three years are insufficient for a robust population- or climate-ensemble confidence claim; 24 sparse samples per month cannot assess heatwave duration/persistence.
- This is **historical year-to-year confirmation**, **not a future SSP/CMIP6 climate-shift experiment**. Avoid claims of future-warming transfer or physical downscaling skill.
- Next steps: report calibration/false-positive and high-tail bias by year, use independent weather-block uncertainty, pre-register prospective validation rule, expand to longer/continuous time series and CMIP6 future scenarios. Do not run additional tail-weight hyperparameter fishing on 2015–2017.

## Independently tested result parser

The stand-alone `scripts/report_fresh_tail_confirmation.py` and `tests/test_fresh_tail_report.py` require three arms, matching data SHA, three seeds and exactly 2015–2017 years; both the script and test were committed separately and main-unit-test workflow runs **45755768208**, **45755780076** succeeded.
