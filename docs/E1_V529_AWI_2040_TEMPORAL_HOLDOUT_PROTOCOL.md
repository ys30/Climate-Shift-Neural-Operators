# E1 v5.29 predeclared 2040 temporal holdout — AWI

## Why this is a separate experiment
The first three GCM continuous-event evaluations and Pareto fronts all use 2050 summer. Repeated scrutiny of that same window cannot establish cross-year stability. Before accessing the new 2040 climate records, fix 2040 as the temporal holdout for AWI, SSP245 and SSP585. Use identical June 1–August 30 source and June 2–August 31 target days, strict native calendar and native 26×27 spatial grid, same 91 continuous daily mean TAS predictions and same GCM-Historical-only month-local 95th percentile event rule (>=3 consecutive days at the same grid cell).

## Locked training and benchmark
Reuse the original 18 AWI Historical-only frozen checkpoints trained 1980–2004 with 2005–2014 checkpoint validation, original original source and Historical normalization, same 1×/2×/4× loss arms and three original seeds. Do not choose a winning arm based on 2050 Pareto results and do not tune or fit on 2040 SSP. Compare all six neural arms and persistence, in both SSPs. AWI 2040 is an independent temporal evaluation window, NOT another independent climate model member, and the two SSP pathways remain correlated.

## Important limitations
2040 is fixed after exploration of 2050, so the analysis is a predeclared follow-up holdout rather than a pre-registered original entire project. One model and one simulated member is not a climate ensemble. Native daily mean TAS is not tasmax or observed heatwave risk, and >=3-day same-pixel event-day Recall/Precision is not event tracking. Summer boundary event labels remain censored. If the 2040 results contradict the 2050 results, report that as important scientific evidence rather than dropping the scenario.

Next planned extension: repeat original frozen BCC and CAMS in 2040 then repeat 2080 across all three original GCMs if computational cost permits.
