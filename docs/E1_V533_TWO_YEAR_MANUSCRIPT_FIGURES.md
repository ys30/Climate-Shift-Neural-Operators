# E1 v5.33 three-GCM two-year manuscript figures

Source: only the completed v5.32 original archived 2040/2050 three-GCM comparison (run 46080073542), which verifies matching Historical-only SHA identity across years and 54 frozen network checkpoints. There are 12 correlated GCM/year/SSP evaluation cells and 72 neural arm evaluations.

Figures: (1) 2 SSP × 3 independent GCM facet lines across 2040 and 2050 for native Persistence, original standard FNO 1×, original Historical-tail-weight FNO 4× event-day Recall. (2) All 12 cells' source-normalized next-day weighted RMSE gain of standard FNO versus Persistence, 4×−1× high-TAS event Recall improvement and 4×−1× event Precision change. The single observed exception where Precision did not fall must be labeled in the full Results/CSV, not silently omitted.

Interpretation: Future simulated native daily-mean TAS rather than observed tasmax; same-cell >=3 adjacent above Historical train-only monthly-local P95 event-day labels, not event-object tracking; 91-day summer window boundary censoring; three independent GCM members, two correlated SSPs × two years per GCM. The 2040 follow-up year was chosen after exploratory 2050 work. Do not treat 12 cells, spatial pixels or model seeds as independent GCMs; no p-values or ex-post hyperparameter choice. These figures document descriptive replication only.
