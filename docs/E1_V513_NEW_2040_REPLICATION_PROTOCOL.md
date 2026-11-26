# E1 v5.13 — source-matched 2040 replication audit

## Goal
Compare *already archived* v4.8 AWI, v5.4 BCC and v5.12 new-2040 neural scores. The v5.12 source-year uses seven new dates per month, whereas original tests use two dates per month in 2030/2050/2080. Models and historical training remain completely fixed.

## Comparisons
For each GCM/SSP/architecture/historical-weight arm:
- 2030/2050/2080 original aggregate area-weighted RMSE and new 2040 area-weighted RMSE (Kelvin).
- **Within-cohort relative skill** = 1 - neural RMSE / own native Persistence RMSE; compare these skills descriptively across cohorts rather than assuming old/new absolute RMSE are identically distributed.
- Local historical-trained monthly P95 Recall and Precision, and Recall gap versus Persistence.
- FNO standard-MSE 1× success against Persistence; FNO 4× minus 1× Recall and RMSE tradeoff within the new 2040 cohort.

The 2 GCM × 2 SSP × 2 architecture × 3 training weights factorial yields 24 fixed neural-arm rows and four summary cells. No model selection or tuning from SSP targets.

## Scientific limitations
This is a **supplementary new-year/date consistency check**, not a formal independent prospective replication or full climate-ensemble uncertainty estimate. Different climate years and sampling frequencies affect prevalence and error magnitude. Grid cells and optimization seeds are not independent GCM replicates. Neither 84 noncontinuous target days nor original 72 projected target pairs can quantify heatwave duration or forced future climate trend. Report descriptive signs, values and counterexamples rather than statistical p-values on four correlated GCM×SSP cells.

## Subsequent priority
Increase fully held-out multi-year daily coverage and independent CMIP6 members/GCMs before strong climate-shift generalization statements.
