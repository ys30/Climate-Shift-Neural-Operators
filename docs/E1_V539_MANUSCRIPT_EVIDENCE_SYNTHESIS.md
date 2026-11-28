# E1 v5.39 — Evidence synthesis for manuscript: simulated sustained hot-event fidelity across years and durations

**Status:** results in this note are extracted from verified successful archived GitHub Actions runs, not from synthetic smoke tests. Date: 2026-11-28.

## Source chain (immutable GitHub Actions evidence)

| Analysis | Verified run | Specific evidence |
|---|---|---|
| Three GCM × two summer years, original three-day event scores | [v5.32 #46080073542](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46080073542) | Same 54 Historical-only frozen checkpoint SHA across 2040 and 2050; 12 GCM×year×SSP scenario cells; 72 neural arm evaluations |
| Paired-year manuscript figures | [v5.33 #46084222081](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46084222081) | RMSE/Recall/Precision results, preserves Precision exception |
| Original AWI 2/3/5-day threshold sensitivity | [v5.34 #46109631398](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46109631398) | Original 3-day metrics reproduced |
| Original BCC 2/3/5-day threshold sensitivity | [v5.35 #46117110466](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46117110466) | Native noleap, original 3-day metrics reproduced |
| Original CAMS 2/3/5-day threshold sensitivity | [v5.36 #46117486346](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46117486346) | Original 3-day metrics reproduced |
| Original three-GCM duration audit | [v5.37 #46117981473](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46117981473) | 108 original arm×duration records, 36 correlated duration cases, all original 3-day metrics reproduced |
| Paired duration-change figures/results | [v5.38 #46119929189](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46119929189) | 3 plotted per-GCM 2/3/5-day metrics; 12 signed 5-day minus 2-day comparisons |

**Paper-facing result 1: average-field skill vs sustained-event fidelity.** Across all 12 original GCM×year×SSP 3-day event cases, standard FNO (Historical-only 1× loss) reduced next-day area-weighted daily-mean TAS RMSE relative to native persistence (**12/12**), but achieved lower three-consecutive-day same-pixel high-TAS event-day Recall (**12/12**). Training with 4× Historical high-TAS loss weight increased event-day Recall relative to the 1× FNO (**12/12**), while reducing Precision (**11/12**). The RMSE effect of this tail upweighting was mixed (RMSE higher **6/12** cases).

**Paper-facing result 2: robustness to event-day duration definitions.** Under the same source-specific Historical monthly local P95 and identical frozen checkpoints, the direction of the Persistence–FNO 1× Recall gap held for **12/12** cases at each duration >=2, >=3 and >=5 consecutive above-threshold days. FNO 4× improved Recall over FNO 1× in **12/12** cases at each threshold. FNO 4× Precision declined in **11/12**, **11/12** and **10/12** at minimum durations 2, 3 and 5, respectively. These are correlated repeated assessments on only three independent GCM members, not 36 independent climate models.

**Paper-facing result 3: signed paired duration sensitivity.** For the *same* GCM×year×SSP cell, the Persistence minus FNO 1× Recall gap increased from a minimum two-day event-day definition to a minimum five-day definition in **all 12/12** cases (zero decreases, zero ties, zero undefined). Meanwhile the FNO 4× versus 1× Recall benefit became larger in **9/12** and smaller in **3/12**, with no ties/undefined. These direction counts are sourced directly from the successful v5.38 plotted, paired comparison, not predicted before running the models.

## Suggested manuscript Results (descriptive; avoid inferential overclaim)

> Across three independently simulated CMIP6 model members (AWI-CM-1-1-MR, BCC-CSM2-MR and CAMS-CSM1-0), two summer years (2040 and 2050), and two correlated SSP pathways per model, standard FNO forecasts consistently improved area-weighted next-day daily-mean temperature RMSE relative to persistence, but achieved lower Recall for event-days belonging to runs of at least three consecutive days above model-specific Historical monthly local 95th-percentile thresholds (12/12 GCM–year–scenario evaluations). Reweighting Historical high-temperature training samples (4× vs 1×) increased event-day Recall in every evaluation but decreased Precision in 11/12. A post hoc sensitivity analysis varying the minimum qualifying run length from two to five days reproduced the standard FNO Recall deficit for every evaluation; the Persistence–FNO Recall gap was larger under the five-day definition in 12/12 paired comparisons. The additional Recall gain from high-temperature training weights also increased in most, but not all, paired evaluations (9/12). Because years and SSPs are correlated within each of the three GCM members, these counts characterize descriptive repeatability rather than independent-sample significance.

## Results framing and non-negotiable caveats

1. **Exactly three independent climate-model members.** Treat 12 GCM×year×SSP cells and 36 duration cells as correlated descriptive comparisons. No `p<0.05` or generalized significance claims from treating days, pixels, SSP pathways, years, or neural seeds as independent.
2. **This is not an observed heatwave hazard forecast.** Target is simulated native-grid *daily mean TAS*, not daily maximum TAS, observed meteorological exposure, human heat risk, event-object tracking, or directly predicted heatwave duration.
3. **Event-day detection**, not tracked-event duration. A minimum 2/3/5-day threshold changes the qualifying observed event-day cohort and its Recall denominator. The increase in Recall gap at >=5 days does not mathematically establish a per-event duration underestimate.
4. **Boundary censoring.** All evaluation windows contain 91 contiguous summer target days, so qualifying episodes overlapping day 1 or day 91 can continue beyond the scored window.
5. **Post hoc exploratory sensitivity.** Two-/five-day thresholds and the separate 2040 follow-up were pursued after inspecting earlier 2050/three-day exploratory results. Report this explicitly; do not call the tests preregistered.
6. **Frozen provenance.** There are 54 original Historical-only neural checkpoints across three GCMs (FNO/U-Net, three Historical tail weights, three seeds). The model evaluations do not refit normalization, weights, or P95 thresholds on SSP projections.
7. **Precision exceptions and mixed RMSE cost.** The tail-upweighting Precision penalty does not hold in all cells; neither does an RMSE penalty. Preserve counterexamples, full scenario tables, and original artifact IDs in supplement.

## Recommended next, not yet verified

- Add **event-object diagnostics** measuring observed/predicted spell durations, episode starts/ends, fragmentation and missed-event probabilities with a transparent temporal/spatial matching protocol. This will test whether the event-day phenomenon translates to per-episode duration bias.
- Add new independent CMIP6 GCM members and a holdout future summer selected before inspecting results; publish uncertainty intervals clustered by GCM when enough members are available.
- Stress-test threshold percentile, edge censoring, and minimum event duration without using SSP information to tune frozen training choices.

Links: [v5.38 manuscript figures and paired 2-to-5 day data](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46119929189/artifacts/15254058049), [v5.37 primary duration audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46117981473/artifacts/15252643144).
