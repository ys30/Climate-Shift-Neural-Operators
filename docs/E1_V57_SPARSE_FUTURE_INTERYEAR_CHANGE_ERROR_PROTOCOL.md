# E1 v5.7: Frozen sampled future-year temperature-change-error audit

## Goal and scope

The v5.5 frozen SSP evaluations contain **per-seed area-weighted forecast bias in Kelvin** for each source GCM, scenario and sampled year. This audit reuses the immutable v4.8 AWI and v5.4 BCC score JSON, without reloading projected daily grids or retraining models.

For the *same native GCM, SSP, grid, and matched dates (days 8 and 20 of each month)*, define the date-averaged regional bias for seed `s` and year `y` as
`b(s,y) = weighted_mean(pred(s,y) - simulated_target(y))`.
Then `b(s,2080) - b(s,2030)` equals the **sampled model-minus-target temperature-change error** over those particular dates.

Each sampled year has 24 +1-day forecast pairs; these are not continuous daily sequences. Inter-year differences in sparse samples mix forced change, internal variability and weather sampling. The resulting contrast is **NOT** observed future climate truth, full annual mean warming, independent ensemble trend attribution, or proof of preserved Historical-to-future warming amplitude.

## Frozen audit protocol

1. Accept only the original AWI v4.8 and BCC v5.4 status, Historical data SHA, checkpoint inventory, train-only selection flags and predeclared scenario/year schedule.
2. Require all 24 sampled pairs per year; identical 8th and 20th calendar-day design across years.
3. Take per-seed forecast bias, then difference matched seed indices across years (seeds 11,22,33 in original evaluation order). Persistence has one deterministic bias.
4. Retain both signed mean change error and mean absolute seed change error; **absolute of mean** and **mean of absolutes** are not interchangeable.
5. Report 2 GCMs × 2 SSPs × 3 interyear comparisons × 7 predictor arms = **84 records**. No hypothesis tests on pixels/seeds/year cells.
6. No hyperparameter or checkpoint selection based on SSP scores; no change to frozen v4.8/v5.4 artifacts.

## Manuscript wording restriction

Permitted: *We assessed bias in sparse, matched-calendar-date projected interyear temperature-change predictions using frozen Historical-trained model checkpoints, separately by GCM and SSP.*

Not permitted: *The neural operators preserve 2030–2080 forced warming* or *proved climate-change response fidelity*. A fully designed Historical-to-SSP warming-amplitude experiment requires additional forecast fields/reference-climatology definitions, denser independent sampling and explicit uncertainty treatment.

## Reproduce

Run `pytest tests/test_cmip6_v57_sampled_change.py`. GitHub Actions workflow `e1-cmip6-v57-change.yml` downloads immutable successful AWI v4.8 and BCC v5.4 artifacts and publishes the non-inferential v5.7 JSON.
