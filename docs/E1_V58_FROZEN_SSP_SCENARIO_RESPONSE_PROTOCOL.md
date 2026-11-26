# E1 v5.8: frozen SSP245 vs SSP585 sampled temperature-response bias

## Why this experiment
E1 v5.7 showed how to retrieve the historical-trained models' year-specific future-simulation forecast biases without rerunning models. The v5.8 calculation changes only the dimension of the diagnostic: compare `SSP585 - SSP245` at **the same sampled future year** and matched 8th/20th calendar dates. It does **not** claim to estimate the externally forced climate response.

## Frozen definition
For GCM `g`, year `y`, and predictor `m`, let `b_m(g,s,y)` denote the spatially area-weighted mean of forecast minus target temperature (Kelvin) over 24 daily pairs in each projected scenario `s`.

Then:
```
response_error_m(g,y) = b_m(g,SSP585,y) - b_m(g,SSP245,y)
                      = [mean(pred_585)-mean(pred_245)]
                       -[mean(target_585)-mean(target_245)].
```

Thus the contrast is exact for the saved regional means, **if** both scoring reports share the same area mask/weight definitions within each GCM and the same sampled calendar-day schedule. Neither grid points nor random seeds are independent climate GCM replications.

## Protocol
- Historical-only frozen AWI v4.8 and BCC v5.4 evaluations are immutable inputs.
- Each source retains its native grid, calendar, 3 historical-trained seeds for neural models, and separate normalization and P95.
- 2 GCM × 3 years (2030, 2050, 2080) × 7 predictor arms (Persistence, FNO × 1x/2x/4x, U-Net × 1x/2x/4x) = 42 records.
- Preserve signed response error and absolute mean response error in Kelvin; compare absolute errors with each GCM's persistence baseline descriptively, not inferentially.
- No model/weight selection from future scores; treat all model/weight combinations as fixed exploratory comparisons.
- SSP trajectories and members differ in weather/internal variability. **Sampled SSP difference is not isolated radiative-forcing response**. No historic-future climate sensitivity claims are warranted without dense sampling/ensemble analysis.

## Next true climate-fidelity gate
E1 v5.9 should obtain denser independent future-day samples, historical reference means, and ideally multiple ensemble members. That would permit a well-defined Historical-to-future change in climatological mean, spatial pattern and extreme quantiles with uncertainty, without conflating sparse day-to-day forecasts and climate projection skill.
