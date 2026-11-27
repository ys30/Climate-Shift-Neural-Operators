# E1 v5.28: Descriptive Pareto figures for continuous hot-event manuscript

Input is ONLY the completed v5.27 verified source archive from GitHub Actions run 45951478573. The renderer must reject changed model choices, nonphysical score values, or stored Pareto labels inconsistent with RMSE, Recall, Precision. It must not read raw future SSP Zarr data, train/refit checkpoints, or compare means as independent GCM replicates.

Outputs: (1) 2×3 six-GCM/SSP panel of event Recall versus Precision, original 7 alternatives labeled and Pareto membership; marker sizes loosely inverse RMSE, while native-grid Persistence-relative RMSE information remains in the linked CSV. (2) six-panel scenario frontier membership counts. Full 42 rows in CSV and English Results text. A frontier point is non-dominated, not an objective winner or selected future-tested model.

Study constraints: 3 independent GCM model members × 2 paired SSP pathways; 91 days of 2050 summer native daily-mean TAS per pathway; local Historical-training monthly P95; same-cell 3-day persistent hot indicators; boundary censoring; descriptive mean across three initialization seeds; no significance or future generalization claim. Do not claim the six cells are independent climate-model experiments or that this Pareto screen is a prospectively validated selection rule.
