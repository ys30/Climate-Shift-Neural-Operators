# E1 v5.18 — Predeclared blind CAMS future SSP score gate

## Frozen checkpoint prerequisite
CAMS v5.17 Historical-only run **45902275281** must fully succeed and supply three weight artifacts, each with both architectures and all three seeds (18 unique checkpoint SHA256). The source Historical-only v5.16 run 45900398047 has already succeeded. Require exact source NPZ SHA, historical normalization and P95 from 900 CAMS training examples; reject any mismatched region, calendar, checkpoint architecture or model state.

## Evaluation
New, genuinely independently calibrated CAMS-CSM1-0 member, same native grid and native calendar as its Historical period. SSP245 and SSP585 sampled 8th/20th monthly in 2030/2050/2080, 72 source-target +1 day pairs per scenario. Fixed checkpoints FNO (width 8, four modes, depth 2) and U-Net (width 8), seeds 11/22/33, historical hot weights 0/1/3. Report weighted RMSE/MAE/bias, month-local P95 Recall/Precision, and own persistence baseline, including year-scoped and across-year descriptive means. Do not pick an arm after future evaluation.

## Study constraints
Daily simulated CMIP6 tas differs from ERA5 instantaneous forecasts. Each SSP contributes climate-simulated future values rather than independent future observations. Historical 2005–2014 validation was used to select trained epochs. Repeated SSP scenario cells from a single GCM are correlated, and sparse days do not establish continuous heatwave duration or forced warming retention. Comparisons across AWI, BCC, CAMS are descriptive, with thresholds trained separately on each GCM.

## Execution safeguard
The workflow is ready, but DO NOT trigger until the v5.17 three-weight matrix and all 18 checkpoints are verified successful. On success v5.18 may be triggered via a new file under `experiments/triggers/e1-cmip6-v518-cams-ssp-v1.txt`. This keeps data lineage and prevents partial training outputs from being treated as frozen.
